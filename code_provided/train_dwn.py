import json
import torch
import numpy as np
import torch_dwn as dwn
import os
import string
from pathlib import Path
import tempfile
import subprocess
import shutil
import re
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from torch import nn
from torch.nn.functional import mse_loss
from utils import rdwn, viz
from utils.preprocessing import seed_everything
import train_m2cgen as m2c
from sklearn.model_selection import train_test_split, LeavePGroupsOut
import csv
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
torch.autograd.set_detect_anomaly(True)

print(f"PyTorch Version: {torch.__version__}")
if hasattr(torch.version, 'cuda'):
    print(f"CUDA Version PyTorch was Built With: {torch.version.cuda}")
else:
    print("PyTorch was installed without CUDA support (CPU-only build).")

class TerSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.333333).float() - (x < -0.3333333).float()

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad

class TerSTE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return TerSTEFunction.apply(x)

class TerProd(torch.nn.Module):
    def __init__(self, size, clamp=1):
        super().__init__()
        self.w = torch.nn.Parameter(2 * torch.rand(size) - 1, requires_grad=True)
        self.clamp = clamp
        self.dummy_terprod = True

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.w.clamp_(-self.clamp, self.clamp)
        return x * TerSTEFunction.apply(self.w)

class SumScale(torch.nn.Module):
    def __init__(self, num_outputs=1, scale=1.):
        super().__init__()
        self.num_outputs = num_outputs
        self.dummy_scale = True
        self.scales = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(scale)) for _ in range(num_outputs)
        ])
        if num_outputs > 1:
            self.biases = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor(scale)) for _ in range(num_outputs)
            ])

    def forward(self, x):
        sum_x = x.sum(dim=1)
        
        outputs = []
        for i in range(self.num_outputs):
            scaled_output = sum_x * self.scales[i]

            if self.num_outputs > 1:
                scaled_output = scaled_output + self.biases[i]
            
            outputs.append(scaled_output)
        return tuple(outputs)


class SumScale2(torch.nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale1 = torch.nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.bias1 = torch.nn.Parameter(torch.tensor(scale), requires_grad=True)

        self.scale2 = torch.nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.bias2 = torch.nn.Parameter(torch.tensor(scale), requires_grad=True)

        self.dummy_scale = True

    def forward(self, x):
        #print(x.sum(dim=1))
        sum_x = x.sum(dim=1)
        scaled_output1 = sum_x * self.scale1 + self.bias1
        scaled_output2 = sum_x * self.scale2 + self.bias2
        return tuple([scaled_output1, scaled_output2])




def evaluate(model, x_test, y_test, device='cuda:0'):
    model.eval()
    with torch.no_grad():
        preds = model(x_test.cuda(device))
        preds_cpu = []
        for pred in preds:
            preds_cpu.append(pred.cpu().numpy())  # Move to CPU and convert
        preds = preds_cpu
        mapes = []; rmses = []; maes = [] 
        for i in range(y_test.shape[1]):
            mape = mean_absolute_percentage_error(y_test[:, i], preds[i])
            mapes.append(mape)

            mse = mean_squared_error(y_test[:, i], preds[i])
            rmse = np.sqrt(mse)
            rmses.append(rmse)

            mae = mean_absolute_error(y_test[:, i], preds[i])
            maes.append(mae)

    return mapes, rmses, maes, preds


def train_and_evaluate(x_train, y_train, params, x_test=None, y_test=None, dump=None, finetune=False, model=None):
    x_train = torch.from_numpy(x_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    if x_test is not None and y_test is not None:
        x_test = torch.from_numpy(x_test.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))
    
    if params['feature_wise']:
        x_train = x_train.flatten(start_dim=1)

    thermometer = dwn.DistributiveThermometer(params['num_bits'], feature_wise=params['feature_wise']).fit(x_train)
    x_train = thermometer.binarize(x_train).flatten(start_dim=1)
    
    if x_test is not None:
        x_test = thermometer.binarize(x_test).flatten(start_dim=1)

    print(thermometer.thresholds)

    # Model
    if not finetune:
        if "sv" in str(dump):
            model = nn.Sequential(
                dwn.LUTLayer(x_train.size(1), params['N1'], n=params['n'], mapping=params['mapping'], alpha=params['alpha'], beta=params['beta'], ste=params['ste'], clamp_luts=params['clamp_luts']),
                dwn.LUTLayer(params['N1'], params['N2'], n=params['n'], ste=params['ste']),
                TerProd(params['N2']),
                #TerProd(params['N1']),
                SumScale()
            )
        else:
            model = nn.Sequential(
                dwn.LUTLayer(x_train.size(1), params['N1'], n=params['n'], mapping=params['mapping'], alpha=params['alpha'], beta=params['beta'], ste=params['ste'], clamp_luts=params['clamp_luts']),
                dwn.LUTLayer(params['N1'], params['N2'], n=params['n'], ste=params['ste']),
                TerProd(params['N2']),
                #TerProd(params['N1']),
                SumScale2()
            )
    # Optimizer and scheduler
    model = model.cuda(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['step_size'])
    
    n_samples = x_train.shape[0]

    for epoch in range(params['epochs']):
        model.train()
        permutation = torch.randperm(n_samples)
        
        mape_train = [0] * y_train.shape[1]
        rmse_train = [0] * y_train.shape[1]
        mae_train = [0] * y_train.shape[1]
        total_train = 0

        for i in range(0, n_samples, params['batch_size']):
            optimizer.zero_grad()

            indices = permutation[i:i+params['batch_size']]

            batch_x = x_train[indices].cuda(params['device'])
            loss = 0
            batch_ys = []
            outputs = model(batch_x)

            for n in range(y_train.shape[1]):
                batch_y = y_train[indices, n].cuda(params['device'])
                batch_ys.append(batch_y)
                loss_fn = training_parameters['loss_fn']
                loss += loss_fn(outputs[n], batch_y)

            loss.backward()
            optimizer.step()

            for n in range(y_train.shape[1]):
                mape_train[n] += batch_ys[n].size(0) * mean_absolute_percentage_error(batch_ys[n].cpu().numpy(), outputs[n].cpu().detach().numpy())
                rmse_train[n] += batch_ys[n].size(0) * np.sqrt(mean_squared_error(batch_ys[n].cpu().numpy(), outputs[n].cpu().detach().numpy()))
                mae_train[n] += batch_ys[n].size(0) * mean_absolute_error(batch_ys[n].cpu().numpy(), outputs[n].cpu().detach().numpy())
            total_train += batch_ys[0].size(0)

        for n in range(y_train.shape[1]):
            mape_train[n] = mape_train[n] / total_train
            rmse_train[n] = rmse_train[n] / total_train
            mae_train[n] = mae_train[n] / total_train
            scheduler.step()

        if x_test is not None and y_test is not None:
            test_mape, test_rmse, test_mae, test_pred = evaluate(model, x_test, y_test, params['device'])
            for n in range(y_train.shape[1]):
                print(f"Epoch {epoch + 1}/{params['epochs']}, Train MAPE for y{n+1}: {mape_train[n]:.4f}, Train RMSE for y{n+1}: {rmse_train[n]: .4f}, Train MAE for y{n+1}: {mae_train[n]:.4f}, Test MAPE for y{n+1}: {test_mape[n]:.4f}, Test RMSE for y{n+1}: {test_rmse[n]:.4f}, Test MAE for y{n+1}: {test_mae[n]:.4f}")
        else: # utilizing all offline data to build DWN - nothing to test
            test_mape, test_rmse, test_mae, test_pred = None, None, None, None
            for n in range(y_train.shape[1]):
                print(f"Epoch {epoch + 1}/{params['epochs']}, Train MAPE for y{n+1}: {mape_train[n]:.4f}, Train RMSE for y{n+1}: {rmse_train[n]: .4f}, Train MAE for y{n+1}: {mae_train[n]:.4f}")

    # if x_test is None and y_test is None:  # utilizing all offline data to build DWN - nothing to test
    #     test_mape, test_rmse, test_mae, test_pred = None, None, None, None
    # else:
    #     test_mape, test_rmse, test_mae, test_pred = evaluate(model, x_test, y_test, params['device'])

    # if np.mean(mape_train) > 1:
    #     print("hi")

    if dump:
        if "sv" in str(dump):
            rdwn.dump_model(thermometer.thresholds.cpu(), model.cpu(), Path(dump))
        else:
            rdwn.dump_model2(thermometer.thresholds.cpu(), model.cpu(), Path(dump))
    return model, test_mape, test_rmse, test_mae, test_pred


def dwn2zephyr(input_dir="inputs", output_dir="outputs"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # File pairs to process: (input_file, output_file)
    files_to_convert = [
        ("WNN.c", "WNN.c"),
        ("WNN.h", "WNN.h"), 
        ("BP_WNN.c", "BP_WNN.c"),
        ("BP_WNN.h", "BP_WNN.h")
    ]
    
    for input_file, output_file in files_to_convert:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
            
        print(f"Converting {input_file}...")
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply conversions based on file type
        if input_file.endswith('.c'):
            content = convert_c_file(content, input_file)
        elif input_file.endswith('.h'):
            content = convert_h_file(content, input_file)
        
        # Write converted content to output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Successfully converted {input_file} -> {output_file}")

def convert_c_file(content, filename):
    """Convert C source file from Arduino to Zephyr format."""
    
    # Remove Arduino-specific includes first
    content = content.replace('#include <avr/pgmspace.h>\n', '')
    content = content.replace('#include "run_inference.h"\n', '')
    content = content.replace('#include <stdint.h>\n#include <stdio.h>\n', '')
    content = content.replace('', '')
    content = content.replace('#if OCE_ENABLED\nstatic const int32_t thresholds[]', '#if defined(CONFIG_DWN_SV) || defined(CONFIG_DWN_BP)\n\nstatic const int32_t thresholds[]')
    content = content.replace('#endif\n\nstatic const uint8_t fc_1_map_lo', '\n\nstatic const uint8_t fc_1_map_lo')
    content = content.replace('#if OCE_ENABLED\nstatic uint8_t scratchpad1', 'static uint8_t scratchpad1')
    if '#else' in content:
        lines = content.split('\n')
        new_lines = []
        skip_section = False
        for line in lines:
            if line.strip() == '#else':
                skip_section = True
                continue
            elif line.strip() == '#else' and skip_section:
                continue
            elif line.strip() == '#endif' and skip_section:
                skip_section = False
                continue
            elif not skip_section:
                new_lines.append(line)
        content = '\n'.join(new_lines)

    # Add Zephyr-specific includes at the beginning
    zephyr_includes = '''#include "system.h"
#include "pan_tompkins.h"
#include "user_ble.h"
#include "WNN.h"
#include "BP_WNN.h"
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(''' + filename.replace('.c', '') + ''', LOG_LEVEL_INF);

'''
    content = zephyr_includes + content
    
    # Replace PROGMEM with Zephyr section attributes
    content = content.replace('PROGMEM', '__attribute__((section(".rodata")))')

    # Remove Arduino compatibility section
    if '#ifdef ARDUINO' in content:
        # Remove the entire ARDUINO section
        lines = content.split('\n')
        new_lines = []
        skip_section = False
        for line in lines:
            if line.strip() == '#ifdef ARDUINO':
                skip_section = True
                continue
            elif line.strip() == '#else' and skip_section:
                continue
            elif line.strip() == '#endif' and skip_section:
                skip_section = False
                continue
            elif not skip_section:
                new_lines.append(line)
        content = '\n'.join(new_lines)

    if '/*' in content:
        lines = content.split('\n')
        new_lines = []
        skip_section = False
        for line in lines:
            if line.strip() == '/*':
                skip_section = True
                continue
            elif line.strip() == '#else' and skip_section:
                continue
            elif line.strip() == '*/' and skip_section:
                skip_section = False
                continue
            elif not skip_section:
                new_lines.append(line)
        content = '\n'.join(new_lines)

    if filename == "WNN.c":
        content = content.replace('#if OCE_ENABLED\nint32_t run_encoding_and_inference(const int32_t *buf_in) {', 'int32_t SV_result = 0;\nTwoInts BP_result = {0};\nvoid run_encoding_and_inference(void* data, uint16_t len) {')
        content = content.replace('encode_input(buf_in, scratchpad1);', 'encode_input(WNN_input, scratchpad1);\n\n#ifdef CONFIG_DWN_SV\nSV_result = run_inference(scratchpad1);\nLOG_INF("SV result: %d", SV_result);\n#endif\n\n#ifdef CONFIG_DWN_BP\nBP_result = bp_run_inference(scratchpad1);\nLOG_INF("BP result: %d, %d", BP_result.a, BP_result.b);\n#endif\n\n\n#ifndef CONFIG_RAW_TRI\n\n#if defined(CONFIG_DWN_SV) && defined(CONFIG_DWN_BP)\nsend_SVBP(&SV_result, &BP_result);\n#elif defined(CONFIG_DWN_SV)\nsend_SV(&SV_result);\n#elif defined(CONFIG_DWN_BP)\nsend_BP(&BP_result);\n#endif\n\n#endif')
        content = content.replace('return run_inference(scratchpad1);\n\n}\n#endif', '}\n')
        content = content.replace('uint8_t oce_enabled(void) { return OCE_ENABLED; }', '')
        content = content.replace('int get_fixp_frac_bits(void) { return FIXP_FRAC_BITS; }\n#endif', 'int get_fixp_frac_bits(void) { return FIXP_FRAC_BITS; }\n#endif\n\n#endif')

    if '#if OCE_ENABLED' in content:
        # Remove the entire OCE_ENABLED section
        lines = content.split('\n')
        new_lines = []
        skip_section = False
        for line in lines:
            if line.strip() == '#if OCE_ENABLED':
                skip_section = True
                continue
            elif line.strip() == '#else' and skip_section:
                continue
            elif line.strip() == '#endif' and skip_section:
                skip_section = False
                continue
            elif not skip_section:
                new_lines.append(line)
        content = '\n'.join(new_lines)

    if '#ifndef ARDUINO' in content:
        # Remove the entire ARDUINO section
        lines = content.split('\n')
        new_lines = []
        skip_section = False
        for line in lines:
            if line.strip() == '#ifndef ARDUINO':
                skip_section = True
                continue
            elif line.strip() == '#else' and skip_section:
                continue
            elif line.strip() == '#endif' and skip_section:
                skip_section = False
                continue
            elif not skip_section:
                new_lines.append(line)
        content = '\n'.join(new_lines)

    if 'TwoInts run_encoding_and_inference(const int32_t *buf_in) {' in content:
        # Remove the entire run_encoding_and_inference section if BP_WNN.
        lines = content.split('\n')
        new_lines = []
        skip_section = False
        for line in lines:
            if line.strip() == '#ifndef ARDUINO':
                skip_section = True
                continue
            elif line.strip() == '#else' and skip_section:
                continue
            elif line.strip() == 'return bp_run_inference(scratchpad1);\n\n}' and skip_section:
                skip_section = False
                continue
            elif not skip_section:
                new_lines.append(line)
        content = '\n'.join(new_lines)
    
    # Remove duplicate OCE_ENABLED definitions and clean up
    content = content.replace('#define OCE_ENABLED 1\n\n#ifndef OCE_ENABLED\n#define OCE_ENABLED 0\n#endif\n\n', '')

    if filename == "BP_WNN.c":
        content = content.replace('INPUT_SAMPLE_SIZE', 'BP_INPUT_SAMPLE_SIZE')
        content = content.replace('get_input_sample_size', 'bp_get_input_sample_size')
        content = content.replace('TwoInts run_inference', 'TwoInts bp_run_inference')
        content = content.replace('void encode_input(', 'void bp_encode_input(')
        # Replace function calls
        content = content.replace('encode_input(buf_in, scratchpad1);', 'bp_encode_input(buf_in, scratchpad1);')
        content = content.replace('return run_inference(scratchpad1);', 'return bp_run_inference(scratchpad1);')
        
        # Replace getter function definitions and handle INPUT_SAMPLE_SIZE constant
        content = content.replace('int get_num_inputs(void) { return INPUTS; }', '')
        content = content.replace('int bp_get_input_sample_size(void) { return BP_INPUT_SAMPLE_SIZE; }', 'int bp_get_input_sample_size(void) { return BP_INPUT_SAMPLE_SIZE; }\n\n#endif')
        content = content.replace('uint8_t is_regression(void) { return REGRESSION; }', '')
        content = content.replace('uint8_t oce_enabled(void) { return OCE_ENABLED; }', '')
        content = content.replace('#if REGRESSION\nint get_fixp_frac_bits(void) { return FIXP_FRAC_BITS; }\n#endif', '')
    
    return content


def convert_h_file(content, filename):
    """Convert header file from Arduino to Zephyr format."""
    
    if filename == "WNN.h":
        content = content.replace('_RUN_INFERENCE_H_', '_WNN_H_')
        content = content.replace('#define OCE_ENABLED 1\n\n#ifndef OCE_ENABLED\n#define OCE_ENABLED 1\n#endif\n\n\n#include <stdint.h>', '')
        content = content.replace('#ifdef __cplusplus\nextern "C" {\n#endif', '')
        content = content.replace('#if OCE_ENABLED\nextern void encode_input(const int32_t *buf_in, uint8_t *buf_out);\n#endif', 'extern void encode_input(const int32_t *buf_in, uint8_t *buf_out);')
        content = content.replace('#if OCE_ENABLED\nextern int32_t run_encoding_and_inference(const int32_t *buf_in);\n#endif', 'void run_encoding_and_inference(void* data, uint16_t len);')
        content = content.replace('#ifdef __cplusplus\n}\n#endif', '\n#ifndef ARDUINO\n#define pgm_read_byte(X) (*((uint8_t*)(X)))\n#define pgm_read_word(X) (*((uint16_t*)(X)))\n#define pgm_read_dword(X) (*((uint32_t*)(X)))\n#endif\n\n#ifdef CONFIG_DWN_SV\nextern int32_t SV_result;\n#endif\n#ifdef CONFIG_DWN_BP\nextern TwoInts BP_result;\n#endif\n')
        content = content.replace('\n\n#endif', '\n#endif  // _WNN_H_')
        content = content.replace('\n\n\n\n', '\n\n')
        content = content.replace('\n\n\n', '\n\n')

    elif filename == "BP_WNN.h":
        content = content.replace('_RUN_INFERENCE_H_', '_BP_WNN_H_')
        content = content.replace('#define INPUTS 300', '#if defined(CONFIG_DWN_BP) && !defined(CONFIG_DWN_SV)\n#define INPUTS 300\n#endif\n')
        content = content.replace('#define OCE_ENABLED 1\n\n#ifndef OCE_ENABLED\n#define OCE_ENABLED 1\n#endif\n\n\n#include <stdint.h>', '')
        content = content.replace('INPUT_SAMPLE_SIZE', 'BP_INPUT_SAMPLE_SIZE')
        content = content.replace('#define REGRESSION 1\n#define FIXP_FRAC_BITS 21', '')
        content = content.replace('#ifdef __cplusplus\nextern "C" {\n#endif', '')
        content = content.replace('#if OCE_ENABLED\nextern void encode_input(const int32_t *buf_in, uint8_t *buf_out);\n#endif', '')
        content = content.replace('typedef struct {', '#if defined(CONFIG_DWN_SV) || defined(CONFIG_DWN_BP)\ntypedef struct {')
        content = content.replace('extern TwoInts run_inference(uint8_t *sample);', 'TwoInts bp_run_inference(uint8_t *sample);\n\n#endif')
        content = content.replace('#if OCE_ENABLED\nextern TwoInts run_encoding_and_inference(const int32_t *buf_in);\n#endif', '')
        content = content.replace('extern int get_num_inputs(void);\nextern int get_input_sample_size(void);\nextern uint8_t is_regression(void);\nextern uint8_t oce_enabled(void);\n#if REGRESSION\nextern int get_fixp_frac_bits(void); \n#endif', 'extern int bp_get_input_sample_size(void);')
        content = content.replace('#ifdef __cplusplus\n}\n#endif', '')
        content = content.replace('\n\n\n#endif', '\n#endif  // _BP_WNN_H_')
        content = content.replace('\n\n\n\n', '\n\n')
        content = content.replace('\n\n\n', '\n\n')
    return content


def build_dwn(template_path, context, wd='.'):
    """
    Generates a Makefile from a template file by substituting placeholders.

    Args:
        template_path (Path): The path to the Makefile template.
        output_path (Path): The path where the generated Makefile will be saved.
        context (dict): A dictionary containing variable names and their values.
    """
    template_path = Path(template_path)
    try:
        # Read the template file
        template_content = template_path.read_text()

        # Create a Template object
        makefile_template = string.Template(template_content)

        # Substitute the placeholders with actual values from the context
        # The safe_substitute method avoids a KeyError if a placeholder is missing
        generated_content = makefile_template.safe_substitute(context)

        # Write the new content to the output file
        # output_path.write_text(generated_content)
        # print(f"Successfully generated '{output_path}' from '{template_path}'.")
        with tempfile.NamedTemporaryFile(mode='w+', prefix="Makefile_", delete=False, dir=wd) as tmp_makefile:
            tmp_makefile_path = tmp_makefile.name
            tmp_makefile.write(generated_content)
            tmp_makefile.flush()

        for target in ['clean', 'all']:
            print(f"--- Running 'make {target}' using temporary Makefile: {os.path.basename(tmp_makefile_path)} ---")

            # Construct the make command.
            # The '-f' flag tells 'make' which file to use.
            command = ["make", "-f", tmp_makefile_path, target]

            # Execute the command using subprocess.run
            result = subprocess.run(
                command,
                cwd=wd,              # Run 'make' in the specified directory
                capture_output=True,  # Capture stdConcatenate all participants' offline dataout and stderr
                text=True,            # Decode stdout/stderr as text
                check=True            # Raise an exception if make returns a non-zero exit code
            )

            # Print the output from the make command
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)

        # Define the original and new filenames
        original_c = 'run_inference.c'; original_h = 'run_inference.h'
        new_c = 'WNN.c'; new_h = 'WNN.h'
        subdirectory = context['MODEL'].split('/')[1].split('.json')[0]

        try:
            os.makedirs(os.path.join(wd, subdirectory), exist_ok=True)
            original_c_path = os.path.join(wd, original_c)
            destination_c_path = os.path.join(wd, subdirectory, new_c)
            shutil.move(original_c_path, destination_c_path)
            print(f"Moved '{original_c}' to '{destination_c_path}'")

            original_h_path = os.path.join(wd, original_h)
            destination_h_path = os.path.join(wd, subdirectory, new_h)
            shutil.move(original_h_path, destination_h_path)
            print(f"Moved '{original_h}' to '{destination_h_path}'")

            dwn2zephyr(input_dir=os.path.join(wd, subdirectory), output_dir=os.path.join(wd, subdirectory))

        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Source file '{original_c}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        return True


    except FileNotFoundError:
        print("Error: 'make' command not found. Is it installed and in your PATH?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error: 'make {target}' failed with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    finally:
        # Clean up the temporary file
        if 'tmp_makefile_path' in locals() and os.path.exists(tmp_makefile_path):
            os.remove(tmp_makefile_path)
            print(f"--- Cleaned up temporary Makefile: {os.path.basename(tmp_makefile_path)} ---")


def run_offline_study(target, validation_mode, model_type, training_parameters, Z_within_subj, dataset_input_path, model_path):
    """
    validation_mode can be ["80-20", "LOTO", "LOPO", "LOPO-finetune", "100-0", "global"]. The last two do not provide any metrics. 

    Validation modes:
    80-20: Concatenate all offline data from each subject, randomly shuffle observations, train on 80% and validate on 20%. 
    LOTO: Leave-one-trial-out. Corresponds to leave-one-session-out for 3 subjects for SV. Corresponds to leave-one-trial-out for every subject for BP.
    LOPO: Leave-one-participant-out.
    LOPO-finetune: Pretrain with leave-one-participant-out, but finetune the model on the test participant's data (by LOTO strategy).

    No validation modes: 
    100-0: Concatenate all offline data from each participant, and use it for training without any validation. To get a DWN model for a returning subject in online part of the study.
    Global: Train on all available data across all participants. For initial device deployment on an unseen participant.
    """
    seed_everything() 

    subj_rmses = []; subj_mapes = []; subj_maes = [] 
    altman_subj_id = [] 
    if target.lower() == "bp":
        subjs = ["MJ", "SB", "EW", "PV", "JR", "HS", "JB", "SK"]
        model_path = 'bp_inference_suite'
        altman_gold_sbp = []; altman_test_sbp = []
        altman_gold_dbp = []; altman_test_dbp = []
        if not Z_within_subj:
            datafile_prefix = "CNAP"
        else:
            datafile_prefix = f"CNAP_Z_{Z_within_subj}"
    elif target.lower() == "sv":
        subjs = ["HH", "HS", "HYS", "JB", "PT", "PV", "SK", "SKR", "SS", "TH", "ZL"]
        model_path = 'sv_inference_suite'
        altman_gold_sv = []; altman_test_sv = []
        if not Z_within_subj:
            datafile_prefix = "NICOM"
        else:
            datafile_prefix = f"NICOM_Z_{Z_within_subj}"

    if validation_mode.lower() in ["80-20", "100-0", "loto"]:  # these are participant-specific modes 
        for subj in subjs:
            print(f"Training and evaluating for participant {subj}...")
            filename = os.path.join(dataset_input_path, f'{datafile_prefix}_{subj}.csv')        
            subj_save_dir = f"{subj}_{target.upper()}_{model_type}"
            save_name = f'{subj_save_dir}_{validation_mode}_{model_type}'


            df = pd.read_csv(filename)
            X = np.array(df.iloc[:,:300])

            if target.lower() == "bp":
                ys = np.array(df.iloc[:, 300:302])
                y_choices = df.columns.to_list()[300:302]
            elif target.lower() == "sv":
                ys = np.array(df.iloc[:, 300:301])
                y_choices = df.columns.to_list()[300:301]

            ### 100-0 split. To get a DWN model for a returning subject in online part of the study.
            if validation_mode.lower() == "100-0":
                if model_type.lower() == "dwn":
                    model, _, _, _, _ = train_and_evaluate(x_train=X, 
                                                        y_train=ys, 
                                                        params=training_parameters, 
                                                        dump=os.path.join(model_path, save_name, f'{save_name}.json'))
                    os.makedirs(os.path.join(model_path, save_name), exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_path, save_name, f'{save_name}.pth'))
                    build_context = {
                        "MODEL": os.path.join(save_name, f'{save_name}.json'),
                    }
                    build_dwn(template_path=os.path.join(model_path, 'Makefile'), context=build_context, wd=model_path)
                else:
                    model, _, _, _, _ = m2c.train_m2c_and_evaluate(x_train=X,
                                                           y_train=ys,
                                                           params=training_parameters,
                                                           dump=os.path.join(model_path, save_name, f'{save_name}.c'),
                                                           model=model_type)
                # For 100-0, no predictions or metrics are generated. 

            ### 80-20 split. To evaluate DWN using the 20% validation set.
            x_train, x_test, y_train, y_test = train_test_split(
                    X, ys, test_size=0.2, random_state=42, shuffle=True
                )
            if validation_mode.lower() == "80-20":
                if model_type.lower() == "dwn":
                    model, test_mape, test_rmse, test_mae, test_pred = train_and_evaluate(
                        x_train=x_train,
                        y_train=y_train,
                        params=training_parameters,
                        x_test=x_test,
                        y_test=y_test,
                        dump=False
                    )
                    os.makedirs(os.path.join(model_path, save_name), exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_path, save_name, f'{save_name}.pth'))
                else:
                    model, test_mape, test_rmse, test_mae, test_pred = m2c.train_m2c_and_evaluate(
                        x_train=x_train,
                        y_train=y_train,
                        params=training_parameters,
                        x_test=x_test,
                        y_test=y_test,
                        dump=False, 
                        model=model_type
                    )
            
                # For 80-20, predictions and metrics are generated on the 20% of the concatenated dataset for each participant. 
                subj_rmses.append(test_rmse)
                subj_mapes.append(test_mape)
                subj_maes.append(test_mae)
                if target.lower() == "bp":
                    altman_test_sbp += list(test_pred[0])
                    altman_gold_sbp += list(y_test[:, 0])
                    altman_test_dbp += list(test_pred[1])
                    altman_gold_dbp += list(y_test[:, 1])
                elif target.lower() == 'sv':
                    altman_test_sv += list(test_pred[0])
                    altman_gold_sv += list(y_test[:, 0])
                altman_subj_id += [f'P{subjs.index(subj)+1}'] * list(test_pred[0]).__len__()

            ### Corresponds to leave-one-trial-out for BP and leave-one-session-out for SV. To evaluate DWN using an unseen trial.
            if validation_mode.lower() == "loto":
                trial_ids = df['trial']
                if len(trial_ids.unique()) > 1:
                    trial_rmses = [[], []]; trial_mapes = [[], []]; trial_maes = [[], []]
                    for test_trial in trial_ids.unique():
                        print(f"LOTO Evaluation for patient {subj} : Test on trial {test_trial}")
                        train_mask = trial_ids.isin(trial_ids[trial_ids != test_trial])
                        test_mask = trial_ids.isin([test_trial])
                        x_train, x_test = X[train_mask, :], X[test_mask, :]
                        y_train, y_test = ys[train_mask, :], ys[test_mask, :]

                        # save_name = f'{subj_save_dir}_{validation_mode}_teston_{test_trial}_{model_type}'
                        if model_type.lower() == "dwn":
                            model, test_mape, test_rmse, test_mae, test_pred = train_and_evaluate(
                                x_train=x_train,
                                y_train=y_train,
                                params=training_parameters,
                                x_test=x_test,
                                y_test=y_test,
                                dump=False
                            )
                            os.makedirs(os.path.join(model_path, save_name), exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(model_path, save_name, f'{save_name}.pth'))
                        else: 
                            model, test_mape, test_rmse, test_mae, test_pred = m2c.train_m2c_and_evaluate(
                                x_train=x_train,
                                y_train=y_train,
                                params=training_parameters,
                                x_test=x_test,
                                y_test=y_test,
                                dump=False,
                                model=model_type
                            )

                        for yi in range(ys.shape[1]):
                            trial_rmses[yi].append(test_rmse[yi])
                            trial_mapes[yi].append(test_mape[yi])
                            trial_maes[yi].append(test_mae[yi])

                        if target.lower() == "bp":
                            altman_test_sbp += list(test_pred[0])
                            altman_gold_sbp += list(y_test[:, 0])
                            altman_test_dbp += list(test_pred[1])
                            altman_gold_dbp += list(y_test[:, 1])
                        elif target.lower() == 'sv':
                            altman_test_sv += list(test_pred[0])
                            altman_gold_sv += list(y_test[:, 0])
                        altman_subj_id += [f'P{subjs.index(subj)+1}'] * list(test_pred[0]).__len__()

                    rmses = []; mapes = []; maes = []
                    for y_rmse in trial_rmses: 
                        rmses.append(np.mean(y_rmse))
                    for y_mape in trial_mapes: 
                        mapes.append(np.mean(y_mape))
                    for y_mae in trial_maes: 
                        maes.append(np.mean(y_mae))

                    # For LOTO, predictions and metrics are generated for each participant, which in itself is an average of all trials for that participant.       
                    subj_rmses.append(rmses)
                    subj_mapes.append(mapes)
                    subj_maes.append(maes)
                else: # cases where only 1 session/trial is available, like the majority of SV participants.
                    subj_rmses.append(np.nan)
                    subj_mapes.append(np.nan)
                    subj_maes.append(np.nan)

    if validation_mode.lower() in ["lopo", "lopo-finetune", "global"]:  # these use the global dataset
        filename = os.path.join(dataset_input_path, f'{datafile_prefix}.csv')       
        df = pd.read_csv(filename)
        X = np.array(df.iloc[:,:300])

        if target.lower() == "bp":
            ys = np.array(df.iloc[:, 300:302])
            y_choices = df.columns.to_list()[300:302]
        elif target.lower() == "sv":
            ys = np.array(df.iloc[:, 300:301])
            y_choices = df.columns.to_list()[300:301]
    
        if validation_mode.lower() == "lopo":
            x_train, x_test, y_train, y_test = train_test_split(
                X, ys, test_size=0.2, random_state=42, shuffle=True
            )
            
            subject_ids = df['subject']
            for test_subject in subject_ids.unique():
                print(f"Now testing on subject: {test_subject}")
                subj = subjs[test_subject] 
                subj_save_dir = f"teston_{subj}_{target.upper()}_{model_type}"
                save_name = f'{subj_save_dir}_{validation_mode}_{model_type}'

                train_mask = subject_ids.isin(subject_ids[subject_ids != test_subject])
                test_mask = subject_ids.isin([test_subject])
                x_train, x_test = X[train_mask, :], X[test_mask, :]
                y_train, y_test = ys[train_mask, :], ys[test_mask, :]

                # save_name = f'{subj_save_dir}_{validation_mode}_teston_{test_subject}_{model_type}'
                if model_type.lower() == "dwn":
                    model, test_mape, test_rmse, test_mae, test_pred = train_and_evaluate(
                        x_train=x_train,
                        y_train=y_train,
                        params=training_parameters,
                        x_test=x_test,
                        y_test=y_test,
                        dump=False
                    )
                    os.makedirs(os.path.join(model_path, save_name), exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_path, save_name, f'{save_name}.pth'))
                else:
                    model, test_mape, test_rmse, test_mae, test_pred = m2c.train_m2c_and_evaluate(
                        x_train=x_train,
                        y_train=y_train,
                        params=training_parameters,
                        x_test=x_test,
                        y_test=y_test,
                        dump=False,
                        model=model_type
                    )
                    fig, ax = plt.subplots(figsize=(30, 30))
                    xgb.plot_tree(model[0], num_trees=4, ax=ax)
                    plt.savefig('xgboost_tree.png')
   

                # For LOPO, predictions and metrics are generated on the test participant. 
                subj_rmses.append(test_rmse)
                subj_mapes.append(test_mape)
                subj_maes.append(test_mae)

                if target.lower() == "bp":
                    altman_test_sbp += list(test_pred[0])
                    altman_gold_sbp += list(y_test[:, 0])
                    altman_test_dbp += list(test_pred[1])
                    altman_gold_dbp += list(y_test[:, 1])
                elif target.lower() == 'sv':
                    altman_test_sv += list(test_pred[0])
                    altman_gold_sv += list(y_test[:, 0])
                altman_subj_id += [f'P{subjs.index(subj)+1}'] * list(test_pred[0]).__len__()

        if validation_mode.lower() == "lopo-finetune":
            x_train, x_test, y_train, y_test = train_test_split(
                X, ys, test_size=0.2, random_state=42, shuffle=True
            )
            
            subject_ids = df['subject']
            for test_subject in subject_ids.unique():
                subj = subjs[test_subject] 
                if target.lower() == "sv" and (subj not in ["SK", "TH", "ZL"]): # only SK, TH, ZL have multiple sessions
                    subj_rmses.append(np.nan)
                    subj_mapes.append(np.nan)
                    subj_maes.append(np.nan)
                    continue
                subj_save_dir = f"teston_{subj}_{target.upper()}_{model_type}"
                save_name = f'{subj_save_dir}_{validation_mode}_{model_type}'

                train_mask = subject_ids.isin(subject_ids[subject_ids != test_subject])
                test_mask = subject_ids.isin([test_subject])
                x_train, x_test = X[train_mask, :], X[test_mask, :]
                y_train, y_test = ys[train_mask, :], ys[test_mask, :]

                # save_name = f'{subj_save_dir}_{validation_mode}_teston_{test_subject}_{model_type}'

                # Pretrain the model on the training data, using the LOPO strategy
                print(f"LOPO Evaluation for patient {subj} : Pretraining")
                if model_type.lower() == "dwn":
                    model, test_mape, test_rmse, test_mae, test_pred = train_and_evaluate(
                        x_train=x_train,
                        y_train=y_train,
                        params=training_parameters,
                        x_test=x_test,
                        y_test=y_test,
                        dump=False,
                        finetune=False
                    )

                # Finetune the model on the test subject's data, using the LOTO strategy

                filename = os.path.join(dataset_input_path, f'{datafile_prefix}_{subj}.csv')        
                subj_save_dir = f"teston_{subj}_{target.upper()}_{model_type}"
                finetune_df = pd.read_csv(filename)
                finetune_X = np.array(finetune_df.iloc[:,:300])

                if target.lower() == "bp":
                    finetune_ys = np.array(finetune_df.iloc[:, 300:302])
                    finetune_y_choices = finetune_df.columns.to_list()[300:302]
                elif target.lower() == "sv":
                    finetune_ys = np.array(finetune_df.iloc[:, 300:301])
                    finetune_y_choices = finetune_df.columns.to_list()[300:301]

                trial_ids = finetune_df['trial']
                if len(trial_ids.unique()) > 1:
                    trial_rmses = [[], []]; trial_mapes = [[], []]; trial_maes = [[], []]
                    for test_trial in trial_ids.unique():
                        print(f"{subj} : Finetuning : using trial {test_trial} for evaluation")
                        train_mask = trial_ids.isin(trial_ids[trial_ids != test_trial])
                        test_mask = trial_ids.isin([test_trial])
                        x_train, x_test = finetune_X[train_mask, :], finetune_X[test_mask, :]
                        y_train, y_test = finetune_ys[train_mask, :], finetune_ys[test_mask, :]

                        # save_name = f'{subj_save_dir}_{validation_mode}_teston_{test_trial}_{model_type}'
                        model, test_mape, test_rmse, test_mae, test_pred = train_and_evaluate(
                            x_train=x_train,
                            y_train=y_train,
                            params=finetuning_parameters,
                            x_test=x_test,
                            y_test=y_test,
                            dump=False,
                            finetune=True,
                            model=model
                        )
                        os.makedirs(os.path.join(model_path, save_name), exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(model_path, save_name, f'{save_name}.pth'))

                        for yi in range(finetune_ys.shape[1]):
                            trial_rmses[yi].append(test_rmse[yi])
                            trial_mapes[yi].append(test_mape[yi])
                            trial_maes[yi].append(test_mae[yi])

                        if target.lower() == "bp":
                            altman_test_sbp += list(test_pred[0])
                            altman_gold_sbp += list(y_test[:, 0])
                            altman_test_dbp += list(test_pred[1])
                            altman_gold_dbp += list(y_test[:, 1])
                        elif target.lower() == 'sv':
                            altman_test_sv += list(test_pred[0])
                            altman_gold_sv += list(y_test[:, 0])
                        altman_subj_id += [f'P{subjs.index(subj)+1}'] * list(test_pred[0]).__len__()

                    rmses = []; mapes = []; maes = []
                    for y_rmse in trial_rmses: 
                        rmses.append(np.mean(y_rmse))
                    for y_mape in trial_mapes: 
                        mapes.append(np.mean(y_mape))
                    for y_mae in trial_maes: 
                        maes.append(np.mean(y_mae))
                    # For LOTO, predictions and metrics are generated for each participant, which in itself is an average of all trials for that participant.       
                    subj_rmses.append(rmses)
                    subj_mapes.append(mapes)
                    subj_maes.append(maes)

                else: # cases where only 1 session/trial is available, like the majority of SV participants.
                    print("Skipping finetuning for this subject because only 1 session is available.")
                    subj_rmses.append(np.nan)
                    subj_mapes.append(np.nan)
                    subj_maes.append(np.nan)


        if validation_mode.lower() == "global":
            subj = 'GLOBAL'
            subj_save_dir = f"{subj}_{target.upper()}_{model_type}"
            save_name = f'{subj_save_dir}_{validation_mode}_{model_type}'

            ### To get a DWN model for a returning subject in online part of the study.
            if model_type.lower() == "dwn":
                model, _, _, _, _ = train_and_evaluate(x_train=X, 
                                                    y_train=ys, 
                                                    params=training_parameters, 
                                                    dump=os.path.join(model_path, save_name, f'{save_name}.json'))
                torch.save(model.state_dict(), os.path.join(model_path, save_name, f'{save_name}.pth'))
                build_context = {
                    "MODEL": os.path.join(save_name, f'{save_name}.json'),
                }

                build_dwn(template_path=os.path.join(model_path, 'Makefile'), context=build_context, wd=model_path)
            else:
                model, _, _, _, _ = m2c.train_m2c_and_evaluate(x_train=X,
                                                               y_train=ys,
                                                               params=training_parameters,
                                                               dump=os.path.join(model_path, save_name, f'{save_name}.c'),
                                                               model=model_type)

    ### Write to disk the metrics & predictions from the study. 
    if validation_mode.lower() in ["80-20", "loto", "lopo"]:
        metrics_output_path = os.path.join(model_path, f"{validation_mode}_{Z_within_subj}_metrics.csv")
        with open(metrics_output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            colnames = ["subj"]
            for yc in y_choices:
                colnames.append(f"rmse_{yc}")
            for yc in y_choices:
                colnames.append(f"mape_{yc}")
            for yc in y_choices:
                colnames.append(f"mae_{yc}")

            writer.writerow(colnames)
            for subji, rmses, mapes, maes in zip(subjs, subj_rmses, subj_mapes, subj_maes):
                try:
                    rmses = [x for x in rmses if not np.isnan(x)]
                    mapes = [x for x in mapes if not np.isnan(x)]
                    maes = [x for x in maes if not np.isnan(x)]
                    row = [subji] + list(rmses) + list(mapes) + list(maes)
                    writer.writerow(row)
                except TypeError:
                    print(f"Skipping {subji} because this subject has only 1 session/trial available.")

        predictions_output_path = os.path.join(model_path, f"{validation_mode}_predictions.csv")
        with open(predictions_output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            if target.lower() == 'sv':
                writer.writerow(["subj", "predicted_sv", "actual_sv"])
            elif target.lower() == 'bp':
                writer.writerow(["subj", "predicted_sbp", "actual_sbp", "predicted_dbp", "actual_dbp"])

            for i in range(len(altman_subj_id)):
                if target.lower() == 'sv':
                    writer.writerow([altman_subj_id[i], altman_test_sv[i], altman_gold_sv[i]])
                elif target.lower() == 'bp':
                    writer.writerow([altman_subj_id[i], altman_test_sbp[i], altman_gold_sbp[i], altman_test_dbp[i], altman_gold_dbp[i]])

if __name__ == "__main__":

    # before optimization
    # training_parameters = {
    #     'device': 'cuda:0',
    #     'loss_fn': mse_loss,
    #     'num_bits': 64,
    #     'feature_wise': True,
    #     'N1': 200,
    #     'N2': 1400,
    #     'n': 6,
    #     'lr': 1e-2,
    #     'epochs': 32,
    #     'step_size': 14,
    #     'gamma': 0.1,
    #     'batch_size': 32,
    #     'mapping': 'learnable',
    #     'alpha': None,
    #     'beta': None,
    #     'ste': True,
    #     'clamp_luts': True,  # equal to ste
    # }
    
    # after optimization, generic
    training_parameters = {
        'device': 'cuda:0',
        'loss_fn': mse_loss,
        'num_bits': 5,
        'feature_wise': True,
        'N1': 200, # 160?
        'N2': 600,
        'n': 6,
        'lr': 1e-2,
        'epochs': 50,
        'step_size': 20,
        'gamma': 0.1,
        'batch_size': 32,
        'mapping': 'learnable',
        'alpha': None,
        'beta': None,
        'ste': True,
        'clamp_luts': True,  # equal to ste
    }

    finetuning_parameters = {
        'device': 'cuda:0',
        'loss_fn': mse_loss,
        'num_bits': 5,
        'feature_wise': True,
        'N1': 200,
        'N2': 200,
        'n': 6,
        'lr': 5e-3,
        'epochs': 50,
        'step_size': 14,
        'gamma': 0.1,
        'batch_size': 32,
        'mapping': 'learnable',
        'alpha': None,
        'beta': None,
        'ste': True,
        'clamp_luts': True,  # equal to ste
    }

    xgboost_parameters = {
        'max_depth': 2,
        'n_estimators': 20,
    }
    """
    validation_mode can be ["80-20", "LOTO", "LOPO", "LOPO-finetune", "100-0", "global"]. The last two do not provide any metrics. 

    Validation modes:
    80-20: Concatenate all offline data from each subject, randomly shuffle observations, train on 80% and validate on 20%. 
    LOTO: Leave-one-trial-out. Corresponds to leave-one-session-out for 3 subjects for SV. Corresponds to leave-one-trial-out for every subject for BP.
    LOPO: Leave-one-participant-out.
    LOPO-finetune: Pretrain with leave-one-participant-out, but finetune the model on the test participant's data (by LOTO strategy).

    No validation modes: 
    100-0: Concatenate all offline data from each participant, and use it for training without any validation. To get a DWN model for a returning subject in online part of the study.
    Global: Train on all available data across all participants. For initial device deployment on an unseen participant.
    """

    #run_offline_study(target='bp', validation_mode='LOPO', model_type='dwn', training_parameters=training_parameters,
    #                   Z_within_subj='minmax', 
    #                   dataset_input_path='ML', model_path='bp_inference_suite')

    run_offline_study(target='sv', validation_mode='LOPO', model_type='xgboost', training_parameters=xgboost_parameters,
                      Z_within_subj='minmax', dataset_input_path='ML', model_path='sv_inference_suite')
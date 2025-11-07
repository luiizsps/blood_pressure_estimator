import json
import numpy as np
import os
import random
from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_xgboost_and_evaluate(x_train, y_train, params, x_test=None, y_test=None, dump=None, model='xgboost'):
    """
    Train and evaluate XGBoost model for BP prediction
    
    Args:
        x_train: Training features
        y_train: Training targets (SBP and DBP)
        params: XGBoost parameters dict with 'max_depth' and 'n_estimators'
        x_test: Test features (optional)
        y_test: Test targets (optional)
        dump: Path to save model (optional)
        model: Model type (should be 'xgboost')
    
    Returns:
        model: Trained model
        test_mape: MAPE for each output (if test data provided)
        test_rmse: RMSE for each output (if test data provided)
        test_mae: MAE for each output (if test data provided)
        test_pred: Predictions (if test data provided)
    """
    
    # Create XGBoost model with MultiOutputRegressor for multiple outputs (SBP and DBP)
    base_model = xgb.XGBRegressor(
        max_depth=params.get('max_depth', 3),
        n_estimators=params.get('n_estimators', 100),
        learning_rate=params.get('learning_rate', 0.1),
        random_state=42,
        n_jobs=-1
    )
    
    model = MultiOutputRegressor(base_model)
    
    # Train the model
    print(f"Training XGBoost model with {x_train.shape[0]} samples...")
    model.fit(x_train, y_train)
    
    # Evaluate on test set if provided
    if x_test is not None and y_test is not None:
        predictions = model.predict(x_test)
        
        test_mape = []
        test_rmse = []
        test_mae = []
        
        for i in range(y_test.shape[1]):
            mape = mean_absolute_percentage_error(y_test[:, i], predictions[:, i])
            test_mape.append(mape)
            
            mse = mean_squared_error(y_test[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            test_rmse.append(rmse)
            
            mae = mean_absolute_error(y_test[:, i], predictions[:, i])
            test_mae.append(mae)
        
        # Convert predictions to list format for compatibility
        test_pred = [predictions[:, i] for i in range(predictions.shape[1])]
        
        print(f"Test MAPE: {test_mape}")
        print(f"Test RMSE: {test_rmse}")
        print(f"Test MAE: {test_mae}")
    else:
        test_mape = None
        test_rmse = None
        test_mae = None
        test_pred = None
    
    # Save model if dump path is provided
    if dump:
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        
        # Save as pickle for easy loading later
        import pickle
        dump_path_pkl = dump.replace('.c', '.pkl')
        with open(dump_path_pkl, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {dump_path_pkl}")
        
        # Optionally save as JSON (XGBoost specific format)
        if hasattr(model, 'estimators_'):
            for idx, estimator in enumerate(model.estimators_):
                json_path = dump.replace('.c', f'_output{idx}.json')
                estimator.save_model(json_path)
                print(f"Output {idx} model saved to {json_path}")
    
    return model, test_mape, test_rmse, test_mae, test_pred


def run_offline_study(target, validation_mode, model_type, training_parameters, Z_within_subj, dataset_input_path, model_path):
    """
    validation_mode can be ["80-20", "LOTO", "LOPO", "100-0", "global"]. The last two do not provide any metrics. 

    Validation modes:
    80-20: Concatenate all offline data from each subject, randomly shuffle observations, train on 80% and validate on 20%. 
    LOTO: Leave-one-trial-out. Corresponds to leave-one-trial-out for every subject for BP.
    LOPO: Leave-one-participant-out.

    No validation modes: 
    100-0: Concatenate all offline data from each participant, and use it for training without any validation. To get a model for a returning subject in online part of the study.
    Global: Train on all available data across all participants. For initial device deployment on an unseen participant.
    """
    seed_everything() 

    subj_rmses = []
    subj_mapes = []
    subj_maes = [] 
    altman_subj_id = [] 
    
    if target.lower() != "bp":
        raise ValueError("This script is configured for BP only. Use target='bp'")
    
    subjs = ["MJ", "SB", "EW", "PV", "JR", "HS", "JB", "SK"]
    model_path = model_path if model_path else 'bp_inference_suite'
    altman_gold_sbp = []
    altman_test_sbp = []
    altman_gold_dbp = []
    altman_test_dbp = []
    
    if not Z_within_subj:
        datafile_prefix = "CNAP"
    else:
        datafile_prefix = f"CNAP_Z_{Z_within_subj}"

    if validation_mode.lower() in ["80-20", "100-0", "loto"]:  # these are participant-specific modes 
        for subj in subjs:
            print(f"Training and evaluating for participant {subj}...")
            filename = os.path.join(dataset_input_path, f'{datafile_prefix}_{subj}.csv')        
            subj_save_dir = f"{subj}_{target.upper()}_{model_type}"
            save_name = f'{subj_save_dir}_{validation_mode}_{model_type}'

            df = pd.read_csv(filename)
            X = np.array(df.iloc[:, :300])
            ys = np.array(df.iloc[:, 300:302])
            y_choices = df.columns.to_list()[300:302]

            ### 100-0 split. To get a model for a returning subject in online part of the study.
            if validation_mode.lower() == "100-0":
                model, _, _, _, _ = train_xgboost_and_evaluate(
                    x_train=X,
                    y_train=ys,
                    params=training_parameters,
                    dump=os.path.join(model_path, save_name, f'{save_name}.c'),
                    model=model_type
                )
                # For 100-0, no predictions or metrics are generated. 

            ### 80-20 split. To evaluate model using the 20% validation set.
            elif validation_mode.lower() == "80-20":
                x_train, x_test, y_train, y_test = train_test_split(
                    X, ys, test_size=0.2, random_state=42, shuffle=True
                )
                
                model, test_mape, test_rmse, test_mae, test_pred = train_xgboost_and_evaluate(
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
                altman_test_sbp += list(test_pred[0])
                altman_gold_sbp += list(y_test[:, 0])
                altman_test_dbp += list(test_pred[1])
                altman_gold_dbp += list(y_test[:, 1])
                altman_subj_id += [f'P{subjs.index(subj)+1}'] * list(test_pred[0]).__len__()

            ### Corresponds to leave-one-trial-out for BP. To evaluate model using an unseen trial.
            elif validation_mode.lower() == "loto":
                trial_ids = df['trial']
                if len(trial_ids.unique()) > 1:
                    trial_rmses = [[], []]
                    trial_mapes = [[], []]
                    trial_maes = [[], []]
                    
                    for test_trial in trial_ids.unique():
                        print(f"LOTO Evaluation for patient {subj} : Test on trial {test_trial}")
                        train_mask = trial_ids.isin(trial_ids[trial_ids != test_trial])
                        test_mask = trial_ids.isin([test_trial])
                        x_train, x_test = X[train_mask, :], X[test_mask, :]
                        y_train, y_test = ys[train_mask, :], ys[test_mask, :]

                        model, test_mape, test_rmse, test_mae, test_pred = train_xgboost_and_evaluate(
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

                        altman_test_sbp += list(test_pred[0])
                        altman_gold_sbp += list(y_test[:, 0])
                        altman_test_dbp += list(test_pred[1])
                        altman_gold_dbp += list(y_test[:, 1])
                        altman_subj_id += [f'P{subjs.index(subj)+1}'] * list(test_pred[0]).__len__()

                    rmses = []
                    mapes = []
                    maes = []
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
                else:
                    # cases where only 1 session/trial is available
                    subj_rmses.append(np.nan)
                    subj_mapes.append(np.nan)
                    subj_maes.append(np.nan)

    if validation_mode.lower() in ["lopo", "global"]:  # these use the global dataset
        filename = os.path.join(dataset_input_path, f'{datafile_prefix}.csv')       
        df = pd.read_csv(filename)
        X = np.array(df.iloc[:, :300])
        ys = np.array(df.iloc[:, 300:302])
        y_choices = df.columns.to_list()[300:302]
    
        if validation_mode.lower() == "lopo":
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

                model, test_mape, test_rmse, test_mae, test_pred = train_xgboost_and_evaluate(
                    x_train=x_train,
                    y_train=y_train,
                    params=training_parameters,
                    x_test=x_test,
                    y_test=y_test,
                    dump=False,
                    model=model_type
                )
                
                # Optional: plot tree structure
                # fig, ax = plt.subplots(figsize=(30, 30))
                # xgb.plot_tree(model[0], num_trees=4, ax=ax)
                # plt.savefig(f'xgboost_tree_{subj}.png')

                # For LOPO, predictions and metrics are generated on the test participant. 
                subj_rmses.append(test_rmse)
                subj_mapes.append(test_mape)
                subj_maes.append(test_mae)

                altman_test_sbp += list(test_pred[0])
                altman_gold_sbp += list(y_test[:, 0])
                altman_test_dbp += list(test_pred[1])
                altman_gold_dbp += list(y_test[:, 1])
                altman_subj_id += [f'P{subjs.index(subj)+1}'] * list(test_pred[0]).__len__()

        elif validation_mode.lower() == "global":
            subj = 'GLOBAL'
            subj_save_dir = f"{subj}_{target.upper()}_{model_type}"
            save_name = f'{subj_save_dir}_{validation_mode}_{model_type}'

            model, _, _, _, _ = train_xgboost_and_evaluate(
                x_train=X,
                y_train=ys,
                params=training_parameters,
                dump=os.path.join(model_path, save_name, f'{save_name}.c'),
                model=model_type
            )

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
            writer.writerow(["subj", "predicted_sbp", "actual_sbp", "predicted_dbp", "actual_dbp"])

            for i in range(len(altman_subj_id)):
                writer.writerow([altman_subj_id[i], altman_test_sbp[i], altman_gold_sbp[i], 
                               altman_test_dbp[i], altman_gold_dbp[i]])


if __name__ == "__main__":
    
    xgboost_parameters = {
        'max_depth': 2,
        'n_estimators': 20,
    }
    
    """
    validation_mode can be ["80-20", "LOTO", "LOPO", "100-0", "global"]. The last two do not provide any metrics. 

    Validation modes:
    80-20: Concatenate all offline data from each subject, randomly shuffle observations, train on 80% and validate on 20%. 
    LOTO: Leave-one-trial-out. Corresponds to leave-one-trial-out for every subject for BP.
    LOPO: Leave-one-participant-out.

    No validation modes: 
    100-0: Concatenate all offline data from each participant, and use it for training without any validation. To get a model for a returning subject in online part of the study.
    Global: Train on all available data across all participants. For initial device deployment on an unseen participant.
    """

    run_offline_study(
        target='bp', 
        validation_mode='LOTO', 
        model_type='xgboost', 
        training_parameters=xgboost_parameters,
        Z_within_subj='minmax', 
        dataset_input_path='ML/subject_data/', 
        model_path='bp_inference_suite'
    )
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import pandas as pd
from pulsedb_loader import load_pulsedb_mimic
from BloodPressureCNNLSTM import BloodPressureCNNLSTM


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = "../../datasets/PulseDB"
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (implicit remainder)

CNN_LAYERS   = [64, 128, 128, 256]
LSTM_LAYERS  = [128, 64]
KERNEL_SIZE  = 5
DROPOUT_RATE = 0.3
EPOCHS       = 10
BATCH_SIZE   = 64


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def split_data(ppg, labels, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Sequential 80 / 10 / 10 split on the concatenated segment array.

    Parameters
    ----------
    ppg    : (N, 1250, 1)
    labels : (N, 2)  — col 0 = SBP, col 1 = DBP

    Returns
    -------
    X_train, X_val, X_test         : (n, 1250, 1)
    y_train, y_val, y_test         : (n, 2)
    """
    n       = len(ppg)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        'train': (0,            n_train),
        'val':   (n_train,      n_train + n_val),
        'test':  (n_train + n_val, n),
    }

    X, y = {}, {}
    for name, (s, e) in splits.items():
        X[name] = ppg[s:e]
        y[name] = labels[s:e]
        print(f"  {name.capitalize():6s}: {e - s:>6,} segments")

    return (X['train'], X['val'], X['test'],
            y['train'], y['val'], y['test'])


def train_single_target(X_train, y_train, X_val, y_val, X_test, y_test,
                         target='SBP'):
    """
    Build, train, and evaluate one BloodPressureCNNLSTM model.

    Parameters
    ----------
    X_*    : (n, timesteps, 1)
    y_*    : (n, 1)  — single BP target column
    target : 'SBP' or 'DBP'
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING MODEL FOR {target}")
    print(f"{'='*70}")

    input_shape = (X_train.shape[1], X_train.shape[2])   # (1250, 1)
    model = BloodPressureCNNLSTM(input_shape=input_shape, target=target)

    model.build_model(
        cnn_layers=CNN_LAYERS,
        lstm_layers=LSTM_LAYERS,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE,
    )
    print(model.model.summary())

    history = model.train_model(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        epochs=EPOCHS,   batch_size=BATCH_SIZE,
        verbose=1,
    )

    results = model.evaluate(X_test=X_test, y_test=y_test, plot=True)

    results_df = pd.DataFrame([results])
    out_csv    = f"bp_prediction_results_{target.lower()}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Results saved → {out_csv}")

    return model, history, results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    ppg, labels = load_pulsedb_mimic(DATA_DIR)
    # ppg    : (N, 1250, 1)
    # labels : (N, 2)  — col 0 = SBP, col 1 = DBP

    # 2. Split ────────────────────────────────────────────────────────────────
    print("\nSplitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(ppg, labels)

    # 3. Separate SBP / DBP targets ───────────────────────────────────────────
    y_sbp_train, y_dbp_train = y_train[:, 0:1], y_train[:, 1:2]
    y_sbp_val,   y_dbp_val   = y_val[:,   0:1], y_val[:,   1:2]
    y_sbp_test,  y_dbp_test  = y_test[:,  0:1], y_test[:,  1:2]

    # 4. Train SBP ────────────────────────────────────────────────────────────
    model_sbp, history_sbp, results_sbp = train_single_target(
        X_train, y_sbp_train,
        X_val,   y_sbp_val,
        X_test,  y_sbp_test,
        target='SBP',
    )

    # 5. Train DBP ────────────────────────────────────────────────────────────
    model_dbp, history_dbp, results_dbp = train_single_target(
        X_train, y_dbp_train,
        X_val,   y_dbp_val,
        X_test,  y_dbp_test,
        target='DBP',
    )

    # 6. Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"  SBP  MAE: {results_sbp['mae']:.2f} mmHg  |  "
          f"RMSE: {results_sbp['rmse']:.2f}  |  "
          f"AAMI: {results_sbp['aami_pass']}  |  BHS: {results_sbp['bhs_grade']}")
    print(f"  DBP  MAE: {results_dbp['mae']:.2f} mmHg  |  "
          f"RMSE: {results_dbp['rmse']:.2f}  |  "
          f"AAMI: {results_dbp['aami_pass']}  |  BHS: {results_dbp['bhs_grade']}")
    print("="*70)

    return {
        'sbp': {'model': model_sbp, 'history': history_sbp, 'results': results_sbp},
        'dbp': {'model': model_dbp, 'history': history_dbp, 'results': results_dbp},
    }


if __name__ == "__main__":
    results = main()
import pandas as pd
import os
import numpy as np
from scipy.signal import cheby2, butter, filtfilt, find_peaks
from BloodPressureCNNLSTM import BloodPressureCNNLSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ===============================
# DATA LOADING
# ===============================

def load_data(path):
    """
    Loads the MIMIC-BP dataset (PPG + ECG), subject-wise split.
    Split:
        80% train
        10% validation
        10% test
    """
    if not os.path.exists(path):
        raise ValueError("Wrong data path")

    path_ppg = os.path.join(path, "ppg/")
    path_ecg = os.path.join(path, "ecg/")
    path_labels = os.path.join(path, "labels/")

    subjects = sorted(os.listdir(path_ppg))
    n_subjects = len(subjects)

    n_train = int(0.8 * n_subjects)
    n_val = int(0.1 * n_subjects)

    train_files = subjects[:n_train]
    val_files = subjects[n_train:n_train + n_val]
    test_files = subjects[n_train + n_val:]

    train_df = load_subjects(train_files, path_ppg, path_ecg, path_labels)
    val_df = load_subjects(val_files, path_ppg, path_ecg, path_labels)
    test_df = load_subjects(test_files, path_ppg, path_ecg, path_labels)

    return train_df, val_df, test_df


def load_subjects(file_list, path_ppg, path_ecg, path_labels):
    data = {"ppg": [], "ecg": [], "labels": []}

    fs = 125
    segment_length = 5 * fs  # 625 samples

    for fname in file_list:
        ppg = np.load(os.path.join(path_ppg, fname))
        ecg = np.load(os.path.join(path_ecg, fname[:7] + "_ecg.npy"))
        labels = np.load(os.path.join(path_labels, fname[:7] + "_labels.npy"))

        ppg = ppg[:, -segment_length:]
        ecg = ecg[:, -segment_length:]

        data["ppg"].append(ppg)
        data["ecg"].append(ecg)
        data["labels"].append(labels)

    return explode_segments(pd.DataFrame(data))


def explode_segments(df):
    rows = []
    for ppg_arr, ecg_arr, label_arr in zip(df.ppg, df.ecg, df.labels):
        for i in range(ppg_arr.shape[0]):
            rows.append({
                "ppg": ppg_arr[i],
                "ecg": ecg_arr[i],
                "label": label_arr[i]
            })
    return pd.DataFrame(rows)


# ===============================
# FEATURE PREPARATION
# ===============================

def extract_rr_intervals(ecg_signals, fs=125):
    """
    Extract R-R interval durations from ECG signals.
    
    Parameters:
    -----------
    ecg_signals : array, shape (n_samples, timesteps)
        ECG signals
    fs : int
        Sampling frequency
    
    Returns:
    --------
    rr_intervals : array, shape (n_samples, 2)
        Two R-R interval durations for each sample (in seconds)
    """
    n_samples = ecg_signals.shape[0]
    rr_intervals = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        ecg = ecg_signals[i]
        
        # Find R peaks using simple peak detection
        # You might want to use Pan-Tompkins algorithm for better accuracy
        peaks, _ = find_peaks(ecg, distance=int(0.5*fs), height=np.mean(ecg))
        
        if len(peaks) >= 3:
            # Calculate first two R-R intervals in seconds
            rr_intervals[i, 0] = (peaks[1] - peaks[0]) / fs
            rr_intervals[i, 1] = (peaks[2] - peaks[1]) / fs
        elif len(peaks) == 2:
            # Only one R-R interval available
            rr_intervals[i, 0] = (peaks[1] - peaks[0]) / fs
            rr_intervals[i, 1] = rr_intervals[i, 0]  # Duplicate
        else:
            # No valid peaks found, use default value
            rr_intervals[i, 0] = 0.8  # Typical RR interval (~75 bpm)
            rr_intervals[i, 1] = 0.8
    
    return rr_intervals


def prepare_features_and_targets(df):
    """
    Prepare features and targets, including R-R interval extraction.
    """
    X_ppg = np.stack(df["ppg"].values)  # (n, 625)
    X_ecg = np.stack(df["ecg"].values)  # (n, 625)

    X = np.stack([X_ppg, X_ecg], axis=-1)  # (n, 625, 2)

    y = np.stack(df["label"].values)
    y_sbp = y[:, 0:1]
    y_dbp = y[:, 1:2]
    
    # Extract R-R intervals from ECG
    print("  Extracting R-R intervals from ECG...")
    rr_intervals = extract_rr_intervals(X_ecg)

    print(f"X shape: {X.shape}")
    print(f"R-R intervals shape: {rr_intervals.shape}")
    print(f"SBP shape: {y_sbp.shape}")
    print(f"DBP shape: {y_dbp.shape}")

    return X, rr_intervals, y_sbp, y_dbp


# ===============================
# SIGNAL PROCESSING
# ===============================

def bandpass_filter(X, fs=125, signal_type="ppg", order=4, rs=20):
    """
    Band-pass filtering for physiological signals.

    PPG:
        - Chebyshev Type II
        - 0.5–10 Hz

    ECG:
        - Butterworth
        - 0.5–40 Hz

    Zero-phase filtering is applied using filtfilt.
    """
    nyq = 0.5 * fs

    if signal_type.lower() == "ppg":
        # PPG → Chebyshev Type II, 0.5–10 Hz
        low, high = 0.5 / nyq, 10.0 / nyq
        b, a = cheby2(order, rs, [low, high], btype="bandpass")

    elif signal_type.lower() == "ecg":
        # ECG → Butterworth, 0.5–40 Hz
        low, high = 0.5 / nyq, 40.0 / nyq
        b, a = butter(order, [low, high], btype="bandpass")

    else:
        raise ValueError("signal_type must be 'ppg' or 'ecg'")

    X_filt = np.zeros_like(X)

    for i in range(X.shape[0]):
        for ch in range(X.shape[2]):
            X_filt[i, :, ch] = filtfilt(b, a, X[i, :, ch])

    return X_filt


def normalize_labels_single(y_train, y_val, y_test):
    """
    Normalize labels using MinMaxScaler for a single target.
    
    Returns:
    --------
    Normalized labels and the scaler
    """
    scaler_y = MinMaxScaler()
    y_train_n = scaler_y.fit_transform(y_train)
    y_val_n = scaler_y.transform(y_val)
    y_test_n = scaler_y.transform(y_test)
    
    return y_train_n, y_val_n, y_test_n, scaler_y


# ===============================
# TRAINING SINGLE TARGET MODEL
# ===============================

def train_single_target(X_train, rr_train, y_train,
                       X_val, rr_val, y_val,
                       X_test, rr_test, y_test,
                       target='SBP'):
    """
    Train and evaluate a single-target model (SBP or DBP).
    
    Parameters:
    -----------
    X_train, X_val, X_test : PPG + ECG signals
    rr_train, rr_val, rr_test : R-R intervals
    y_train, y_val, y_test : BP labels for single target
    target : str, 'SBP' or 'DBP'
    
    Returns:
    --------
    model, history, results
    """
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL FOR {target}")
    print(f"{'='*70}")
    
    # Normalize labels
    print(f"\nNormalizing {target} labels...")
    y_train_norm, y_val_norm, y_test_norm, scaler_y = normalize_labels_single(
        y_train, y_val, y_test
    )
    
    # Initialize model
    print(f"\nInitializing CNN-LSTM model for {target}...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (625, 2)
    model = BloodPressureCNNLSTM(input_shape=input_shape, target=target)
    
    # Attach the scaler to the model
    model.scaler_y = scaler_y
    
    # Normalize signals using model's per-channel scalers
    print(f"\nNormalizing signals with per-channel scalers...")
    X_train_norm = model.normalize_signals(X_train, fit=True)
    X_val_norm = model.normalize_signals(X_val, fit=False)
    X_test_norm = model.normalize_signals(X_test, fit=False)
    
    # Build model architecture
    print("\nBuilding model architecture...")
    cnn_layers = [64, 128, 128, 256]
    lstm_layers = [128, 64]
    
    model.build_model(
        cnn_layers=cnn_layers,
        lstm_layers=lstm_layers,
        kernel_size=5,
        dropout_rate=0.3
    )
    
    # Print model summary
    print(model.model.summary())
    
    # Train model with normalized labels and R-R intervals
    print(f"\nTraining {target} model...")
    history = model.train_model(
        X_train=X_train_norm,
        rr_train=rr_train,
        y_train=y_train_norm,
        X_val=X_val_norm,
        rr_val=rr_val,
        y_val=y_val_norm,
        epochs=100,
        batch_size=64,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\nEvaluating {target} model on test set...")
    results = model.evaluate(
        X_test=X_test_norm,
        rr_test=rr_test,
        y_test=y_test_norm,
        plot=True
    )
    
    # Save results to CSV
    print(f"\nSaving {target} results to CSV...")
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'bp_prediction_results_{target.lower()}.csv', index=False)
    print(f"Results saved to 'bp_prediction_results_{target.lower()}.csv'")
    
    print(f"\n{target} training complete!")
    
    return model, history, results


# ===============================
# MAIN
# ===============================

def main():
    print("Loading data...")
    train_df, val_df, test_df = load_data("../MIMIC_III")

    print("\nPreparing datasets...")
    X_train, rr_train, y_sbp_train, y_dbp_train = prepare_features_and_targets(train_df)
    X_val, rr_val, y_sbp_val, y_dbp_val = prepare_features_and_targets(val_df)
    X_test, rr_test, y_sbp_test, y_dbp_test = prepare_features_and_targets(test_df)

    print("\nBandpass filtering ECG and PPG...")

    # --- PPG filtering (channel 0) ---
    X_train_ppg = bandpass_filter(
        X_train[:, :, [0]], fs=125, signal_type="ppg"
    )
    X_val_ppg = bandpass_filter(
        X_val[:, :, [0]], fs=125, signal_type="ppg"
    )
    X_test_ppg = bandpass_filter(
        X_test[:, :, [0]], fs=125, signal_type="ppg"
    )

    # --- ECG filtering (channel 1) ---
    X_train_ecg = bandpass_filter(
        X_train[:, :, [1]], fs=125, signal_type="ecg"
    )
    X_val_ecg = bandpass_filter(
        X_val[:, :, [1]], fs=125, signal_type="ecg"
    )
    X_test_ecg = bandpass_filter(
        X_test[:, :, [1]], fs=125, signal_type="ecg"
    )

    # Recombine channels (PPG, ECG)
    X_train = np.concatenate([X_train_ppg, X_train_ecg], axis=2)
    X_val = np.concatenate([X_val_ppg, X_val_ecg], axis=2)
    X_test = np.concatenate([X_test_ppg, X_test_ecg], axis=2)

    # Train SBP model
    model_sbp, history_sbp, results_sbp = train_single_target(
        X_train, rr_train, y_sbp_train,
        X_val, rr_val, y_sbp_val,
        X_test, rr_test, y_sbp_test,
        target='SBP'
    )
    
    # Train DBP model
    model_dbp, history_dbp, results_dbp = train_single_target(
        X_train, rr_train, y_dbp_train,
        X_val, rr_val, y_dbp_val,
        X_test, rr_test, y_dbp_test,
        target='DBP'
    )
    
    # Print summary of both models
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\nSBP Model:")
    print(f"  MAE: {results_sbp['mae']:.2f} mmHg")
    print(f"  STD: {results_sbp['std']:.2f} mmHg")
    print(f"  r: {results_sbp['r']:.3f}")
    print(f"  AAMI/IEEE: {results_sbp['aami_pass']}")
    print(f"  BHS Grade: {results_sbp['bhs_grade']}")
    
    print(f"\nDBP Model:")
    print(f"  MAE: {results_dbp['mae']:.2f} mmHg")
    print(f"  STD: {results_dbp['std']:.2f} mmHg")
    print(f"  r: {results_dbp['r']:.3f}")
    print(f"  AAMI/IEEE: {results_dbp['aami_pass']}")
    print(f"  BHS Grade: {results_dbp['bhs_grade']}")
    print("="*70)
    
    print("\nAll training complete!")

    return {
        'sbp': {'model': model_sbp, 'history': history_sbp, 'results': results_sbp},
        'dbp': {'model': model_dbp, 'history': history_dbp, 'results': results_dbp}
    }


if __name__ == "__main__":
    results = main()
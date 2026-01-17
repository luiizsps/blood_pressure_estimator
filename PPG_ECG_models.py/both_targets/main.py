import pandas as pd
import os
import numpy as np
from scipy.signal import cheby2, butter, filtfilt
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

def prepare_features_and_targets(df):
    X_ppg = np.stack(df["ppg"].values)  # (n, 625)
    X_ecg = np.stack(df["ecg"].values)  # (n, 625)

    X = np.stack([X_ppg, X_ecg], axis=-1)  # (n, 625, 2)

    y = np.stack(df["label"].values)
    y_sbp = y[:, 0:1]
    y_dbp = y[:, 1:2]

    print(f"X shape: {X.shape}")
    print(f"SBP shape: {y_sbp.shape}")
    print(f"DBP shape: {y_dbp.shape}")

    return X, y_sbp, y_dbp


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

    Parameters
    ----------
    X : ndarray
        Shape (n_samples, timesteps, channels)
    fs : int
        Sampling frequency (Hz)
    signal_type : str
        'ppg' or 'ecg'
    order : int
        Filter order
    rs : float
        Stopband attenuation (dB), used only for Chebyshev II

    Returns
    -------
    X_filt : ndarray
        Filtered signal, same shape as X
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


def normalize_labels(y_sbp_train, y_dbp_train,
                     y_sbp_val, y_dbp_val,
                     y_sbp_test, y_dbp_test):
    """
    Normalize labels using MinMaxScaler fitted on training data.
    
    Returns:
    --------
    Normalized labels and the scaler
    """
    y_train = np.hstack([y_sbp_train, y_dbp_train])
    y_val = np.hstack([y_sbp_val, y_dbp_val])
    y_test = np.hstack([y_sbp_test, y_dbp_test])

    scaler_y = MinMaxScaler()
    y_train_n = scaler_y.fit_transform(y_train)
    y_val_n = scaler_y.transform(y_val)
    y_test_n = scaler_y.transform(y_test)

    return (
        y_train_n[:, 0:1], y_train_n[:, 1:2],
        y_val_n[:, 0:1], y_val_n[:, 1:2],
        y_test_n[:, 0:1], y_test_n[:, 1:2],
        scaler_y
    )


# ===============================
# MAIN
# ===============================

def main():
    print("Loading data...")
    train_df, val_df, test_df = load_data("../MIMIC_III")

    print("\nPreparing datasets...")
    X_train, y_sbp_train, y_dbp_train = prepare_features_and_targets(train_df)
    X_val, y_sbp_val, y_dbp_val = prepare_features_and_targets(val_df)
    X_test, y_sbp_test, y_dbp_test = prepare_features_and_targets(test_df)

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

    print("\nInitializing model...")
    model = BloodPressureCNNLSTM(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    print("\nNormalizing signals with per-channel scalers...")
    # Fit scalers on training data and transform
    X_train_norm = model.normalize_signals(X_train, fit=True)
    # Use fitted scalers for validation and test
    X_val_norm = model.normalize_signals(X_val, fit=False)
    X_test_norm = model.normalize_signals(X_test, fit=False)

    print("\nNormalizing labels...")
    (y_sbp_train_n, y_dbp_train_n,
     y_sbp_val_n, y_dbp_val_n,
     y_sbp_test_n, y_dbp_test_n,
     scaler_y) = normalize_labels(
        y_sbp_train, y_dbp_train,
        y_sbp_val, y_dbp_val,
        y_sbp_test, y_dbp_test
    )
    
    # Attach the label scaler to the model
    model.scaler_y = scaler_y

    print("\nBuilding model architecture...")
    cnn_layers = [64, 32, 16, 16]
    lstm_layers = [32]

    model.build_model(
        cnn_layers=cnn_layers,
        lstm_layers=lstm_layers,
        kernel_size=5,
        dropout_rate=0.3
    )

    print(model.model.summary())

    print("\nTraining model...")
    history = model.train_model(
        X_train=X_train_norm,
        y_train=(y_sbp_train_n, y_dbp_train_n),
        X_val=X_val_norm,
        y_val=(y_sbp_val_n, y_dbp_val_n),
        epochs=100,
        batch_size=64
    )

    print("\nEvaluating model...")
    results = model.evaluate(
        X_test=X_test_norm,
        y_test=(y_sbp_test_n, y_dbp_test_n),
        plot=True
    )

    pd.DataFrame([results]).to_csv("bp_prediction_results.csv", index=False)
    print("\nTraining complete!")

    return model, history, results


if __name__ == "__main__":
    model, history, results = main()
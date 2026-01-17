import pandas as pd
import os
import numpy as np
from scipy.signal import cheby2, filtfilt
from BloodPressureCNNLSTM import BloodPressureCNNLSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    """Loads the MIMIC-BP dataset (PPG only), subject-wise split.
    Each file corresponds to one patient.
    Split:
        80% train
        10% validation
        10% test
    """
    if not os.path.exists(path):
        raise ValueError("Wrong data path")
    
    path_ppg = os.path.join(path, "ppg/")
    path_labels = os.path.join(path, "labels/")
    ppg_files = sorted(os.listdir(path_ppg))
    label_files = sorted(os.listdir(path_labels))
    n_subjects = len(ppg_files)
    n_train = int(0.8 * n_subjects)
    n_val = int(0.1 * n_subjects)
    train_files = ppg_files[:n_train]
    val_files = ppg_files[n_train:n_train + n_val]
    test_files = ppg_files[n_train + n_val:]
    train_df = load_subjects(train_files, path_ppg, path_labels)
    val_df = load_subjects(val_files, path_ppg, path_labels)
    test_df = load_subjects(test_files, path_ppg, path_labels)

    return train_df, val_df, test_df

def load_subjects(ppg_list, path_ppg, path_labels):
    data = {"ppg": [], "labels": []}
    for ppg_file in ppg_list:
        label_file = ppg_file[:7]+"_labels.npy"
        ppg_data = np.load(os.path.join(path_ppg, ppg_file))
        label_data = np.load(os.path.join(path_labels, label_file))
        
        # Extract last 5 seconds from each segment
        fs = 125
        segment_length = 5 * fs  # 625 samples (last 5 seconds)
        ppg_data_5s = ppg_data[:, -segment_length:]  # Take last 625 samples
        
        data["ppg"].append(ppg_data_5s)
        data["labels"].append(label_data)

    return explode_segments(pd.DataFrame(data))

def explode_segments(df):
    """Explode subject arrays into individual segments"""
    rows = []
    for ppg_arr, label_arr in zip(df.ppg, df.labels):
        for seg in range(ppg_arr.shape[0]):
            rows.append({
                "ppg": ppg_arr[seg],
                "label": label_arr[seg]
            })
    return pd.DataFrame(rows)

def prepare_features_and_targets(df):
    """Prepare PPG and labels for CNN1D"""
    X = np.stack(df["ppg"].values)      # (n_samples, 625) - now 5 seconds
    X = X[..., np.newaxis]              # (n_samples, 625, 1)
    y = np.stack(df["label"].values)    # (n_samples, 2)
    y_sbp = y[:, 0:1]
    y_dbp = y[:, 1:2]
    print(f"X shape: {X.shape}")
    print(f"SBP shape: {y_sbp.shape}")
    print(f"DBP shape: {y_dbp.shape}")

    return X, y_sbp, y_dbp

def normalize_ppg(X_train, X_val, X_test):
    """Normalize PPG signals using MinMaxScaler per sample.
    
    Parameters:
    -----------
    X_train : array, shape (n_samples, timesteps, channels)
    X_val : array, shape (n_samples, timesteps, channels)
    X_test : array, shape (n_samples, timesteps, channels)
    
    Returns:
    --------
    X_train_norm, X_val_norm, X_test_norm : normalized arrays
    """
    # Normalize each sample independently (per timestep across all samples would be wrong)
    # We use z-score normalization per sample
    def normalize_per_sample(X):
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mean) / std
    
    X_train_norm = normalize_per_sample(X_train)
    X_val_norm = normalize_per_sample(X_val)
    X_test_norm = normalize_per_sample(X_test)
    
    return X_train_norm, X_val_norm, X_test_norm

def normalize_labels_single(y_train, y_val, y_test):
    """Normalize BP labels using MinMaxScaler for a single target.
    
    Parameters:
    -----------
    y_train : training labels for single target
    y_val : validation labels for single target
    y_test : test labels for single target
    
    Returns:
    --------
    Normalized labels and the scaler object
    """
    # Create and fit scaler on training data only
    scaler_y = MinMaxScaler()
    y_train_norm = scaler_y.fit_transform(y_train)
    y_val_norm = scaler_y.transform(y_val)
    y_test_norm = scaler_y.transform(y_test)
    
    return y_train_norm, y_val_norm, y_test_norm, scaler_y

def bandpass_ppg(X, fs=125, low=0.5, high=10, order=4, rs=20):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq

    b, a = cheby2(order, rs, [low, high], btype='bandpass')

    X_filt = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_filt[i, :, 0] = filtfilt(b, a, X[i, :, 0])

    return X_filt

def train_single_target(X_train, y_train, X_val, y_val, X_test, y_test, target='SBP'):
    """Train and evaluate model for a single target (SBP or DBP).
    
    Parameters:
    -----------
    X_train, X_val, X_test : PPG signals
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
    input_shape = (X_train.shape[1], X_train.shape[2])  # (625, 1)
    model = BloodPressureCNNLSTM(input_shape=input_shape, target=target)
    
    # Attach the scaler to the model
    model.scaler_y = scaler_y
    
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
    
    # Train model with normalized labels
    print(f"\nTraining {target} model...")
    history = model.train_model(
        X_train=X_train,
        y_train=y_train_norm,
        X_val=X_val,
        y_val=y_val_norm,
        epochs=100,
        batch_size=64,
        verbose=1
    )
    
    # Evaluate on test set with normalized labels
    print(f"\nEvaluating {target} model on test set...")
    results = model.evaluate(
        X_test=X_test,
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

def main():
    # Load and prepare data
    print("Loading data...")
    train_df, val_df, test_df = load_data("../MIMIC_III")
    
    print("\nPreparing training data...")
    X_train, y_sbp_train, y_dbp_train = prepare_features_and_targets(train_df)
    
    print("\nPreparing validation data...")
    X_val, y_sbp_val, y_dbp_val = prepare_features_and_targets(val_df)
    
    print("\nPreparing test data...")
    X_test, y_sbp_test, y_dbp_test = prepare_features_and_targets(test_df)

    # Bandpass filter
    print("\nApplying bandpass filter to PPG signals...")
    fs = 125
    X_train = bandpass_ppg(X_train, fs)
    X_val = bandpass_ppg(X_val, fs)
    X_test = bandpass_ppg(X_test, fs)
    
    # Normalize PPG signals
    print("\nNormalizing PPG signals...")
    X_train, X_val, X_test = normalize_ppg(X_train, X_val, X_test)
    
    # Train SBP model
    model_sbp, history_sbp, results_sbp = train_single_target(
        X_train, y_sbp_train,
        X_val, y_sbp_val,
        X_test, y_sbp_test,
        target='SBP'
    )
    
    # Train DBP model
    model_dbp, history_dbp, results_dbp = train_single_target(
        X_train, y_dbp_train,
        X_val, y_dbp_val,
        X_test, y_dbp_test,
        target='DBP'
    )
    
    # Print summary of both models
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\nSBP Model:")
    print(f"  MAE: {results_sbp['mae']:.2f} mmHg")
    print(f"  AAMI/IEEE: {results_sbp['aami_pass']}")
    print(f"  BHS Grade: {results_sbp['bhs_grade']}")
    
    print(f"\nDBP Model:")
    print(f"  MAE: {results_dbp['mae']:.2f} mmHg")
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
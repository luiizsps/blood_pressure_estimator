import os
import h5py
import numpy as np


def deref(f, data):
    """
    Dereferences HDF5 object references if needed.
    MATLAB v7.3 struct arrays store each field as an array of references
    that need to be resolved against the file object.
    """
    data = np.array(data).flatten()
    # If the array contains references, resolve each one
    if len(data) > 0 and isinstance(data[0], h5py.h5r.Reference):
        resolved = [np.array(f[ref]).flatten() for ref in data]
        return np.concatenate(resolved).astype(np.float32)
    return data.astype(np.float32)


def deref_signals(f, data):
    """
    Dereferences PPG_F signal references.
    Each reference points to a (1250,) array for one segment.
    Returns shape (N, 1250).
    """
    data = np.array(data).flatten()
    if len(data) > 0 and isinstance(data[0], h5py.h5r.Reference):
        segments = [np.array(f[ref]).flatten() for ref in data]
        return np.stack(segments, axis=0).astype(np.float32)  # (N, 1250)
    # Not references — fall back to direct array
    arr = np.array(data)
    if arr.ndim == 2:
        return arr.T.astype(np.float32)   # (1250, N) → (N, 1250)
    return arr.astype(np.float32)


def load_pulsedb_mimic(data_dir):
    """
    Loads all MIMIC subject .mat files from data_dir.

    Returns
    -------
    ppg    : np.ndarray, shape (N, 1250, 1)  – PPG signals, ready for CNN-LSTM input
    labels : np.ndarray, shape (N, 2)        – [[SBP, DBP], ...] in mmHg
    """
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {data_dir}")

    all_ppg, all_labels = [], []

    for i, fname in enumerate(mat_files):
        fpath = os.path.join(data_dir, fname)
        try:
            with h5py.File(fpath, 'r') as f:
                wins = f['Subj_Wins']

                sbp = deref(f, wins['SegSBP'])       # (N,)
                dbp = deref(f, wins['SegDBP'])       # (N,)
                ppg = deref_signals(f, wins['PPG_F']) # (N, 1250)

            # Ensure 2D: single-segment files may yield (1250,) instead of (N, 1250)
            if ppg.ndim == 1:
                ppg = ppg[np.newaxis, :]   # (1250,) → (1, 1250)

            all_ppg.append(ppg)
            all_labels.append(np.stack([sbp, dbp], axis=1))
            print(f"[{i+1}/{len(mat_files)}] {fname}  →  {len(sbp)} segments")

        except Exception as e:
            print(f"[!] Skipping {fname}: {e}")

    ppg_all    = np.concatenate(all_ppg,    axis=0)  # (N, 1250)
    labels_all = np.concatenate(all_labels, axis=0)  # (N, 2)

    # Add channel dim → (N, 1250, 1) for CNN-LSTM input
    ppg_all = ppg_all[:, :, np.newaxis]

    print(f"\nDone.")
    print(f"  PPG shape    : {ppg_all.shape}     (samples, timesteps, channels)")
    print(f"  Labels shape : {labels_all.shape}  (samples, [SBP, DBP])")

    return ppg_all, labels_all


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    DATA_DIR = 'Segment_Files/PulseDB_MIMIC'   # ← change to your path

    ppg, labels = load_pulsedb_mimic(DATA_DIR)

    # ppg    → your model input,  shape (N, 1250, 1)
    # labels → your model target, shape (N, 2)  — column 0 = SBP, column 1 = DBP

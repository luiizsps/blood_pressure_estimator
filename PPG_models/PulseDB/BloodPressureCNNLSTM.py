import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class BloodPressureCNNLSTM:
    """
    CNN-LSTM hybrid model for predicting SBP or DBP from PPG signals.
    Expects pre-processed (already filtered and scaled) inputs.
    """

    def __init__(self, input_shape, target='SBP'):
        """
        Parameters
        ----------
        input_shape : tuple
            Shape of a single input sample (timesteps, channels).
        target : str
            'SBP' or 'DBP'.
        """
        self.input_shape = input_shape
        self.target = target.upper()
        if self.target not in ['SBP', 'DBP']:
            raise ValueError("target must be either 'SBP' or 'DBP'")

        self.model   = None
        self.history = None

    def build_model(self, cnn_layers, lstm_layers, kernel_size=5, dropout_rate=0.3):
        """
        Build CNN-LSTM architecture.

        Parameters
        ----------
        cnn_layers   : list of int  – filters per Conv1D layer
        lstm_layers  : list of int  – units per LSTM layer
        kernel_size  : int          – Conv1D kernel size
        dropout_rate : float        – dropout / recurrent-dropout rate
        """
        inputs = layers.Input(shape=self.input_shape, name='ppg_input')
        x = inputs

        x = layers.BatchNormalization(name='bn_input')(x)

        for i, filters in enumerate(cnn_layers):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            if i % 2 == 0:
                x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)

        x = layers.BatchNormalization(name='bn_conv')(x)
        x = layers.Dropout(dropout_rate, name='dropout_conv')(x)

        for i, units in enumerate(lstm_layers):
            return_sequences = (i < len(lstm_layers) - 1)
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=0,
                name=f'lstm_{i+1}'
            )(x)

        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        output = layers.Dense(1, name=f'{self.target.lower()}_output')(x)

        self.model = models.Model(
            inputs=inputs, outputs=output, name=f'BP_CNN_LSTM_{self.target}'
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae', 'mse']
        )

        return self.model

    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                epochs=100, batch_size=64, verbose=1):
        """
        Train using a numpy generator to avoid copying full arrays to GPU.
        Only one batch lives on GPU at a time.
        """

        def make_generator(X, y, shuffle=False):
            """Returns a callable that yields (x_batch, y_batch) numpy arrays."""
            n = len(X)
            indices = np.arange(n)

            def generator():
                idx = indices.copy()
                if shuffle:
                    np.random.shuffle(idx)
                for start in range(0, n, batch_size):
                    batch_idx = idx[start : start + batch_size]
                    yield (
                        X[batch_idx].astype('float32'),
                        y[batch_idx].astype('float32'),
                    )

            return generator

        output_sig = (
            tf.TensorSpec(shape=(None,) + self.input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1),                  dtype=tf.float32),
        )

        train_ds = tf.data.Dataset.from_generator(
            make_generator(X_train, y_train, shuffle=True),
            output_signature=output_sig,
        ).prefetch(tf.data.AUTOTUNE)

        val_ds = None
        if X_val is not None and y_val is not None:
            val_ds = tf.data.Dataset.from_generator(
                make_generator(X_val, y_val, shuffle=False),
                output_signature=output_sig,
            ).prefetch(tf.data.AUTOTUNE)

        callbacks = [
            EarlyStopping(
                monitor='val_loss' if val_ds else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_bp_model_{self.target.lower()}.keras',
                monitor='val_mae' if val_ds else 'mae',
                mode='min',
                save_best_only=True,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def predict(self, X):
        """Return raw model predictions."""
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test, plot=False):
        """
        Evaluate model according to AAMI, BHS, and IEEE standards.

        Parameters
        ----------
        X_test : array (n_samples, timesteps, channels)
        y_test : array (n_samples, 1) – true BP values in mmHg
        plot   : bool – whether to generate scatter + Bland-Altman plots

        Returns
        -------
        dict with all evaluation metrics
        """
        y_pred = self.predict(X_test).flatten()
        y_true = np.array(y_test).flatten()

        print(f"\nTrue {self.target} range: [{y_true.min():.1f}, {y_true.max():.1f}] mmHg")
        print(f"Pred {self.target} range: [{y_pred.min():.1f}, {y_pred.max():.1f}] mmHg")

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        std  = np.std(y_true - y_pred)
        r2   = r2_score(y_true, y_pred)
        r    = np.corrcoef(y_true, y_pred)[0, 1]
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        errors_pct      = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
        mean_error_pct  = np.mean(errors_pct)
        std_error_pct   = np.std(errors_pct)

        print("\n" + "="*70)
        print(f"BLOOD PRESSURE PREDICTION RESULTS - {self.target}")
        print("="*70)
        print(f"  MAE:            {mae:.2f} mmHg")
        print(f"  RMSE:           {rmse:.2f} mmHg")
        print(f"  STD:            {std:.2f} mmHg")
        print(f"  R²:             {r2:.4f}")
        print(f"  r:              {r:.4f}")
        print(f"  MAPE:           {mape:.2f}%")
        print(f"  Mean Error %:   {mean_error_pct:.2f}%")
        print(f"  STD Error %:    {std_error_pct:.2f}%")

        aami_pass = "PASS ✓" if (mae < 5 and std < 8) else "FAIL ✗"
        print(f"  AAMI/IEEE:      {aami_pass}")

        if mae <= 5:
            bhs_grade = "Grade A"
        elif mae <= 10:
            bhs_grade = "Grade B"
        elif mae <= 15:
            bhs_grade = "Grade C"
        else:
            bhs_grade = "Grade D"
        print(f"  BHS Grade:      {bhs_grade}")
        print("="*70)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
            axes[0].plot([y_true.min(), y_true.max()],
                         [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0].set_xlabel(f'True {self.target} (mmHg)', fontsize=12)
            axes[0].set_ylabel(f'Predicted {self.target} (mmHg)', fontsize=12)
            axes[0].set_title(f'{self.target}: MAE={mae:.2f}, r={r:.3f}', fontsize=12)
            axes[0].grid(True, alpha=0.3)

            mean_bp = (y_true + y_pred) / 2
            diff_bp = y_true - y_pred
            axes[1].scatter(mean_bp, diff_bp, alpha=0.5, s=20)
            axes[1].axhline(y=0,          color='r', linestyle='-',  lw=2)
            axes[1].axhline(y= 1.96*std,  color='r', linestyle='--', lw=1)
            axes[1].axhline(y=-1.96*std,  color='r', linestyle='--', lw=1)
            axes[1].set_xlabel(f'Mean {self.target} (mmHg)', fontsize=12)
            axes[1].set_ylabel('Difference (mmHg)', fontsize=12)
            axes[1].set_title(f'Bland-Altman Plot ({self.target})', fontsize=12)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_filename = f'bp_prediction_plots_{self.target.lower()}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"\nPlots saved to '{plot_filename}'")
            plt.show()

        return {
            'target':            self.target,
            'mae':               mae,
            'rmse':              rmse,
            'std':               std,
            'r2':                r2,
            'r':                 r,
            'mape':              mape,
            'mean_error_percent': mean_error_pct,
            'std_error_percent':  std_error_pct,
            'aami_pass':         aami_pass,
            'bhs_grade':         bhs_grade,
        }
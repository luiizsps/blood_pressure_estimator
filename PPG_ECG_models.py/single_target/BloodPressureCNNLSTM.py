import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BloodPressureCNNLSTM:
    """
    CNN-LSTM hybrid model for predicting EITHER SBP OR DBP from PPG, ECG, and R-R intervals.
    This is a single-target model (separate models for SBP and DBP).
    """
    def __init__(self, input_shape, target='SBP'):
        """
        Initialize BP prediction model for a single target.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input (timesteps, channels)
            Example: (625, 2) for PPG + ECG
        target : str
            Target to predict: 'SBP' or 'DBP'
        """
        self.input_shape = input_shape
        self.target = target.upper()
        if self.target not in ['SBP', 'DBP']:
            raise ValueError("target must be 'SBP' or 'DBP'")
        
        self.n_channels = input_shape[1]
        # Create separate scaler for each channel (e.g., PPG and ECG)
        self.scaler_x = [MinMaxScaler() for _ in range(self.n_channels)]
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.history = None
    
    def build_model(self, cnn_layers, lstm_layers, kernel_size=5, dropout_rate=0.3):
        """
        Build CNN-LSTM architecture for single target prediction (SBP OR DBP).
        Incorporates R-R interval features explicitly.
        
        Parameters:
        -----------
        cnn_layers : list
            Number of filters for each CNN layer
        lstm_layers : list
            Number of units for each LSTM layer
        kernel_size : int
            Kernel size for convolutional layers
        dropout_rate : float
            Dropout rate for regularization
        """
        # Input layers
        signal_input = layers.Input(shape=self.input_shape, name='signal_input')
        rr_input = layers.Input(shape=(2,), name='rr_intervals_input')  # Two R-R intervals
        
        x = signal_input
        
        # Batch normalization
        x = layers.BatchNormalization(name='bn_input')(x)
        
        # CNN layers
        for i, filters in enumerate(cnn_layers):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            if i % 2 == 0:  # Max pooling every 2 layers
                x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
        
        # Batch normalization after CNN
        x = layers.BatchNormalization(name='bn_conv')(x)
        x = layers.Dropout(dropout_rate, name='dropout_conv')(x)
        
        # LSTM layers
        for i, units in enumerate(lstm_layers):
            return_sequences = (i < len(lstm_layers) - 1)
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=0,
                name=f'lstm_{i+1}'
            )(x)
        
        # Concatenate LSTM output with R-R intervals
        x = layers.Concatenate(name='concat_rr')([x, rr_input])
        
        # Dense layers for target BP
        x = layers.Dense(64, activation='relu', name=f'{self.target.lower()}_dense_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
        x = layers.Dense(32, activation='relu', name=f'{self.target.lower()}_dense_2')(x)
        output = layers.Dense(1, name=f'{self.target.lower()}_output')(x)
        
        # Compile model
        self.model = models.Model(
            inputs=[signal_input, rr_input], 
            outputs=output, 
            name=f'BP_CNN_LSTM_{self.target}'
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae']
        )
        
        return self.model
    
    def normalize_signals(self, X, fit=False):
        """
        Normalize signals using separate scalers for each channel.
        
        Parameters:
        -----------
        X : array, shape (n_samples, timesteps, channels)
            Input signals (e.g., PPG + ECG)
        fit : bool
            If True, fit the scalers on the data. Use True for training data only.
        
        Returns:
        --------
        X_norm : array, normalized signals with same shape as input
        """
        X_norm = np.zeros_like(X)
        
        for ch in range(self.n_channels):
            # Extract channel
            X_ch = X[:, :, ch]
            n_samples, n_timesteps = X_ch.shape
            
            # Reshape to 2D for scaler (samples * timesteps, 1)
            X_ch_flat = X_ch.reshape(-1, 1)
            
            if fit:
                # Fit and transform
                X_ch_norm = self.scaler_x[ch].fit_transform(X_ch_flat)
            else:
                # Only transform
                X_ch_norm = self.scaler_x[ch].transform(X_ch_flat)
            
            # Reshape back to original shape
            X_norm[:, :, ch] = X_ch_norm.reshape(n_samples, n_timesteps)
        
        return X_norm
    
    def denormalize_signals(self, X):
        """
        Denormalize signals using the fitted scalers.
        
        Parameters:
        -----------
        X : array, shape (n_samples, timesteps, channels)
            Normalized signals
        
        Returns:
        --------
        X_denorm : array, denormalized signals with same shape as input
        """
        X_denorm = np.zeros_like(X)
        
        for ch in range(self.n_channels):
            X_ch = X[:, :, ch]
            n_samples, n_timesteps = X_ch.shape
            X_ch_flat = X_ch.reshape(-1, 1)
            X_ch_denorm = self.scaler_x[ch].inverse_transform(X_ch_flat)
            X_denorm[:, :, ch] = X_ch_denorm.reshape(n_samples, n_timesteps)
        
        return X_denorm
    
    def train_model(self, X_train, rr_train, y_train, X_val=None, rr_val=None, y_val=None, 
                    epochs=100, batch_size=64, verbose=1):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : array, shape (n_samples, timesteps, channels)
            Training signals (PPG + ECG)
        rr_train : array, shape (n_samples, 2)
            R-R interval durations for training
        y_train : array, shape (n_samples, 1)
            Training BP values (normalized)
        X_val : array, optional
            Validation signals
        rr_val : array, optional
            Validation R-R intervals
        y_val : array, optional
            Validation BP values (normalized)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
        """
        # Validation data
        validation_data = None
        if X_val is not None and rr_val is not None and y_val is not None:
            validation_data = ([X_val, rr_val], y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_bp_model_{self.target.lower()}.keras',
                monitor='val_loss' if validation_data else 'loss',
                mode='min',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            [X_train, rr_train],
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X, rr_intervals):
        """Make predictions"""
        predictions = self.model.predict([X, rr_intervals], verbose=0)
        return predictions
    
    def evaluate(self, X_test, rr_test, y_test, plot=True):
        """
        Evaluate model according to AAMI, BHS, and IEEE standards.
        Denormalizes predictions before calculating metrics.
        
        Parameters:
        -----------
        X_test : array
            Test signals (PPG + ECG)
        rr_test : array, shape (n_samples, 2)
            Test R-R intervals
        y_test : array, shape (n_samples, 1)
            Test BP values (normalized)
        plot : bool
            Whether to generate plots
        
        Returns:
        --------
        dict : Dictionary containing all evaluation metrics
        """
        # Get predictions (normalized)
        y_pred_norm = self.predict(X_test, rr_test)
        y_true_norm = y_test
        
        # DENORMALIZE: Convert back to original mmHg scale
        print("\nDenormalizing predictions and true values...")
        
        # Apply inverse transform
        y_pred_original = self.scaler_y.inverse_transform(y_pred_norm)
        y_true_original = self.scaler_y.inverse_transform(y_true_norm)
        
        # Flatten arrays
        y_pred = y_pred_original.flatten()
        y_true = y_true_original.flatten()
        
        print(f"True {self.target} range: [{y_true.min():.1f}, {y_true.max():.1f}] mmHg")
        print(f"Pred {self.target} range: [{y_pred.min():.1f}, {y_pred.max():.1f}] mmHg")
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        std = np.std(y_true - y_pred)
        r2 = r2_score(y_true, y_pred)
        r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        errors_percent = np.abs((y_true - y_pred) / y_true) * 100
        mean_error_percent = np.mean(errors_percent)
        std_error_percent = np.std(errors_percent)
        
        # Print results
        print("\n" + "="*70)
        print(f"{self.target} PREDICTION RESULTS (PPG + ECG + R-R Intervals)")
        print("="*70)
        print(f"  MAE: {mae:.2f} mmHg")
        print(f"  RMSE: {rmse:.2f} mmHg")
        print(f"  STD: {std:.2f} mmHg")
        print(f"  R²: {r2:.4f}")
        print(f"  r: {r:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Mean Error %: {mean_error_percent:.2f}%")
        print(f"  STD Error %: {std_error_percent:.2f}%")
        
        # AAMI/IEEE compliance
        aami = "PASS ✓" if (mae < 5 and std < 8) else "FAIL ✗"
        print(f"  AAMI/IEEE Standard: {aami}")
        
        # BHS Grade
        if mae <= 5:
            bhs = "Grade A"
        elif mae <= 10:
            bhs = "Grade B"
        elif mae <= 15:
            bhs = "Grade C"
        else:
            bhs = "Grade D"
        print(f"  BHS Grade: {bhs}")
        print("="*70)
        
        # Plotting
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Scatter plot
            axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
            axes[0].plot([y_true.min(), y_true.max()], 
                        [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0].set_xlabel(f'True {self.target} (mmHg)', fontsize=12)
            axes[0].set_ylabel(f'Predicted {self.target} (mmHg)', fontsize=12)
            axes[0].set_title(f'{self.target}: MAE={mae:.2f}, r={r:.3f}', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Bland-Altman plot
            mean_bp = (y_true + y_pred) / 2
            diff_bp = y_true - y_pred
            axes[1].scatter(mean_bp, diff_bp, alpha=0.5, s=20)
            axes[1].axhline(y=0, color='r', linestyle='-', lw=2)
            axes[1].axhline(y=1.96*std, color='r', linestyle='--', lw=1)
            axes[1].axhline(y=-1.96*std, color='r', linestyle='--', lw=1)
            axes[1].set_xlabel(f'Mean {self.target} (mmHg)', fontsize=12)
            axes[1].set_ylabel('Difference (mmHg)', fontsize=12)
            axes[1].set_title(f'Bland-Altman Plot ({self.target})', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'bp_prediction_{self.target.lower()}.png', dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to 'bp_prediction_{self.target.lower()}.png'")
            plt.show()
        
        return {
            'mae': mae,
            'rmse': rmse,
            'std': std,
            'r2': r2,
            'r': r,
            'mape': mape,
            'mean_error_percent': mean_error_percent,
            'std_error_percent': std_error_percent,
            'aami_pass': aami,
            'bhs_grade': bhs
        }
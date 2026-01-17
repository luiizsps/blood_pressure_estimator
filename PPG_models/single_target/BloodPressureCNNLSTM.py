import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Forçar uso da CPU (desabilita CuDNN)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class BloodPressureCNNLSTM:
    """
    CNN-LSTM hybrid model for predicting SBP or DBP from PPG signals.
    """
    def __init__(self, input_shape, target='SBP'):
        """
        Initialize BP prediction model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input (timesteps, channels)
        target : str
            Target to predict: 'SBP' or 'DBP'
        """
        self.input_shape = input_shape
        self.target = target.upper()
        if self.target not in ['SBP', 'DBP']:
            raise ValueError("target must be either 'SBP' or 'DBP'")
        
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.history = None
    
    def build_model(self, cnn_layers, lstm_layers, kernel_size=5, dropout_rate=0.3):
        """
        Build CNN-LSTM architecture based on literature best practices.
        
        Architecture follows:
        - Multiple Conv1D layers for spatial feature extraction
        - MaxPooling for dimensionality reduction
        - LSTM layers for temporal modeling
        - Dense layers for regression
        - Single output for either SBP or DBP
        
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
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='ppg_input')
        x = inputs
        
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
        
        # Dense layers for output
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        output = layers.Dense(1, name=f'{self.target.lower()}_output')(x)
        
        # Compile model
        self.model = models.Model(inputs=inputs, outputs=output, name=f'BP_CNN_LSTM_{self.target}')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                    epochs=100, batch_size=64, verbose=1):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : array, shape (n_samples, timesteps, channels)
            Training PPG signals
        y_train : array, shape (n_samples, 1)
            Training BP values (normalized) - either SBP or DBP
        X_val : array, optional
            Validation PPG signals
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
        if (X_val is not None) and (y_val is not None):
            validation_data = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_bp_model_{self.target.lower()}.h5',
                monitor='val_mae' if validation_data else 'mae',
                mode='min',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def evaluate(self, X_test, y_test, plot=False):
        """
        Evaluate model according to AAMI, BHS, and IEEE standards.
        Denormalizes predictions before calculating metrics.
        
        Standards:
        - AAMI: MAE < 5 mmHg, SD < 8 mmHg
        - BHS: Grade A (MAE ≤ 5), Grade B (MAE ≤ 10), Grade C (MAE ≤ 15)
        - IEEE: MAE ≤ 5 mmHg and SD ≤ 8 mmHg
        
        Parameters:
        -----------
        X_test : array
            Test PPG signals
        y_test : array, shape (n_samples, 1)
            Test BP values (normalized) - either SBP or DBP
        plot : bool
            Whether to generate plots
        
        Returns:
        --------
        dict : Dictionary containing all evaluation metrics
        """
        # Get predictions (normalized)
        y_pred_norm = self.predict(X_test)
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
        print(f"BLOOD PRESSURE PREDICTION RESULTS - {self.target}")
        print("="*70)
        print(f"\n{self.target} METRICS:")
        print(f"  MAE: {mae:.2f} mmHg")
        print(f"  RMSE: {rmse:.2f} mmHg")
        print(f"  STD: {std:.2f} mmHg")
        print(f"  R²: {r2:.4f}")
        print(f"  r: {r:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Mean Error %: {mean_error_percent:.2f}%")
        print(f"  STD Error %: {std_error_percent:.2f}%")
        
        # AAMI/IEEE compliance
        aami_pass = "PASS ✓" if (mae < 5 and std < 8) else "FAIL ✗"
        print(f"  AAMI/IEEE Standard: {aami_pass}")
        
        # BHS Grade
        if mae <= 5:
            bhs_grade = "Grade A"
        elif mae <= 10:
            bhs_grade = "Grade B"
        elif mae <= 15:
            bhs_grade = "Grade C"
        else:
            bhs_grade = "Grade D"
        print(f"  BHS Grade: {bhs_grade}")
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
            plot_filename = f'bp_prediction_plots_{self.target.lower()}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"\nPlots saved to '{plot_filename}'")
            plt.show()
        
        return {
            'target': self.target,
            'mae': mae,
            'rmse': rmse,
            'std': std,
            'r2': r2,
            'r': r,
            'mape': mape,
            'mean_error_percent': mean_error_percent,
            'std_error_percent': std_error_percent,
            'aami_pass': aami_pass,
            'bhs_grade': bhs_grade
        }
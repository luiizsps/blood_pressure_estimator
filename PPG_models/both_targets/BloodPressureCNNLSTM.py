import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
import pandas as pd
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Forçar uso da CPU (desabilita CuDNN)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class BloodPressureCNNLSTM:
    """
    CNN-LSTM hybrid model for predicting SBP and DBP from PPG signals.
    """
    def __init__(self, input_shape):
        """
        Initialize BP prediction model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input (timesteps, channels)
        """
        self.input_shape = input_shape
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.history = None
    
    def build_model(self, cnn_layers, lstm_layers, kernel_size=5, dropout_rate=0.3, l2_reg=0.01):
        """
        Build CNN-LSTM architecture based on literature best practices.
        
        Architecture follows:
        - Multiple Conv1D layers for spatial feature extraction
        - MaxPooling for dimensionality reduction
        - LSTM layers for temporal modeling
        - Dense layers for regression
        - Dual output heads for SBP and DBP
        
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
                kernel_regularizer=l2(l2_reg),
                name=f'conv1d_{i+1}'
            )(x)
            
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
                activation="tanh",
                recurrent_dropout=0,
                name=f'lstm_{i+1}'
            )(x)
        
        # Dense layers for SBP
        sbp_branch = layers.Dense(64, activation='relu', name='sbp_dense_1')(x)
        sbp_branch = layers.Dense(32, activation='relu', name='sbp_dense_2')(sbp_branch)
        sbp_output = layers.Dense(1, name='sbp_output')(sbp_branch)
        
        # Dense layers for DBP
        dbp_branch = layers.Dense(64, activation='relu', name='dbp_dense_1')(x)
        dbp_branch = layers.Dense(32, activation='relu', name='dbp_dense_2')(dbp_branch)
        dbp_output = layers.Dense(1, name='dbp_output')(dbp_branch)
        
        outputs = [sbp_output, dbp_output]
        
        # Compile model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='BP_CNN_LSTM')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss={'sbp_output': 'mae', 'dbp_output': 'mae'},
            metrics={'sbp_output': ['mae'], 'dbp_output': ['mae']}
        )
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                    epochs=1000, batch_size=64, verbose=1):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : array, shape (n_samples, timesteps, channels)
            Training PPG signals
        y_train : tuple (y_sbp, y_dbp)
            Training BP values (normalized)
        X_val : array, optional
            Validation PPG signals
        y_val : tuple, optional
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
            validation_data = (X_val, {'sbp_output': y_val[0], 'dbp_output': y_val[1]})
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_bp_model.keras',
                monitor='val_sbp_output_mae',
                mode='min',
                save_best_only=True
            ),
            ReduceLROnPlateau (
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )

        ]
        
        y_train_dict = {'sbp_output': y_train[0], 'dbp_output': y_train[1]}
        
        # Train
        self.history = self.model.fit(
            X_train,
            y_train_dict,
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
    
    def evaluate(self, X_test, y_test, plot=True):
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
        y_test : tuple (y_sbp, y_dbp)
            Test BP values (normalized)
        plot : bool
            Whether to generate plots
        
        Returns:
        --------
        dict : Dictionary containing all evaluation metrics
        """
        # Get predictions (normalized)
        predictions = self.predict(X_test)
        sbp_pred, dbp_pred = predictions
        sbp_true, dbp_true = y_test
        
        # DENORMALIZE: Convert back to original mmHg scale
        # print("\nDenormalizing predictions and true values...")
        
        # # Concatenate for inverse transform
        # y_pred_concat = np.column_stack([sbp_pred_norm, dbp_pred_norm])
        # y_true_concat = np.column_stack([sbp_true_norm, dbp_true_norm])
        
        # # Apply inverse transform
        # y_pred_original = self.scaler_y.inverse_transform(y_pred_concat)
        # y_true_original = self.scaler_y.inverse_transform(y_true_concat)
        
        # # Split back into SBP and DBP (now in mmHg)
        # sbp_pred = y_pred_original[:, 0]
        # dbp_pred = y_pred_original[:, 1]
        # sbp_true = y_true_original[:, 0]
        # dbp_true = y_true_original[:, 1]
        
        print(f"True SBP range: [{sbp_true.min():.1f}, {sbp_true.max():.1f}] mmHg")
        print(f"Pred SBP range: [{sbp_pred.min():.1f}, {sbp_pred.max():.1f}] mmHg")
        print(f"True DBP range: [{dbp_true.min():.1f}, {dbp_true.max():.1f}] mmHg")
        print(f"Pred DBP range: [{dbp_pred.min():.1f}, {dbp_pred.max():.1f}] mmHg")
        
        # Calculate metrics for SBP
        sbp_mae = mean_absolute_error(sbp_true, sbp_pred)
        sbp_rmse = np.sqrt(mean_squared_error(sbp_true, sbp_pred))
        sbp_std = np.std(sbp_true - sbp_pred)
        sbp_r2 = r2_score(sbp_true, sbp_pred)
        sbp_r = np.corrcoef(sbp_true.flatten(), sbp_pred.flatten())[0, 1]
        
        # Calculate percentage errors for SBP
        sbp_mape = np.mean(np.abs((sbp_true - sbp_pred) / sbp_true)) * 100
        sbp_errors_percent = np.abs((sbp_true - sbp_pred) / sbp_true) * 100
        sbp_mean_error_percent = np.mean(sbp_errors_percent)
        sbp_std_error_percent = np.std(sbp_errors_percent)
        
        # Calculate metrics for DBP
        dbp_mae = mean_absolute_error(dbp_true, dbp_pred)
        dbp_rmse = np.sqrt(mean_squared_error(dbp_true, dbp_pred))
        dbp_std = np.std(dbp_true - dbp_pred)
        dbp_r2 = r2_score(dbp_true, dbp_pred)
        dbp_r = np.corrcoef(dbp_true.flatten(), dbp_pred.flatten())[0, 1]
        
        # Calculate percentage errors for DBP
        dbp_mape = np.mean(np.abs((dbp_true - dbp_pred) / dbp_true)) * 100
        dbp_errors_percent = np.abs((dbp_true - dbp_pred) / dbp_true) * 100
        dbp_mean_error_percent = np.mean(dbp_errors_percent)
        dbp_std_error_percent = np.std(dbp_errors_percent)
        
        # Print results
        print("\n" + "="*70)
        print("BLOOD PRESSURE PREDICTION RESULTS")
        print("="*70)
        print("\nSYSTOLIC BLOOD PRESSURE (SBP):")
        print(f"  MAE: {sbp_mae:.2f} mmHg")
        print(f"  RMSE: {sbp_rmse:.2f} mmHg")
        print(f"  STD: {sbp_std:.2f} mmHg")
        print(f"  R²: {sbp_r2:.4f}")
        print(f"  r: {sbp_r:.4f}") 
        print(f"  MAPE: {sbp_mape:.2f}%")
        print(f"  Mean Error %: {sbp_mean_error_percent:.2f}%")
        print(f"  STD Error %: {sbp_std_error_percent:.2f}%")
        
        # AAMI/IEEE compliance for SBP
        sbp_aami = "PASS ✓" if (sbp_mae < 5 and sbp_std < 8) else "FAIL ✗"
        print(f"  AAMI/IEEE Standard: {sbp_aami}")
        
        # BHS Grade for SBP
        if sbp_mae <= 5:
            sbp_bhs = "Grade A"
        elif sbp_mae <= 10:
            sbp_bhs = "Grade B"
        elif sbp_mae <= 15:
            sbp_bhs = "Grade C"
        else:
            sbp_bhs = "Grade D"
        print(f"  BHS Grade: {sbp_bhs}")
        
        print("\nDIASTOLIC BLOOD PRESSURE (DBP):")
        print(f"  MAE: {dbp_mae:.2f} mmHg")
        print(f"  RMSE: {dbp_rmse:.2f} mmHg")
        print(f"  STD: {dbp_std:.2f} mmHg")
        print(f"  R²: {dbp_r2:.4f}")
        print(f"  r: {dbp_r:.4f}")
        print(f"  MAPE: {dbp_mape:.2f}%")
        print(f"  Mean Error %: {dbp_mean_error_percent:.2f}%")
        print(f"  STD Error %: {dbp_std_error_percent:.2f}%")
        
        # AAMI/IEEE compliance for DBP
        dbp_aami = "PASS ✓" if (dbp_mae < 5 and dbp_std < 8) else "FAIL ✗"
        print(f"  AAMI/IEEE Standard: {dbp_aami}")
        
        # BHS Grade for DBP
        if dbp_mae <= 5:
            dbp_bhs = "Grade A"
        elif dbp_mae <= 10:
            dbp_bhs = "Grade B"
        elif dbp_mae <= 15:
            dbp_bhs = "Grade C"
        else:
            dbp_bhs = "Grade D"
        print(f"  BHS Grade: {dbp_bhs}")
        print("="*70)
        
        # Plotting
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # SBP scatter plot
            axes[0, 0].scatter(sbp_true, sbp_pred, alpha=0.5, s=20)
            axes[0, 0].plot([sbp_true.min(), sbp_true.max()], 
                           [sbp_true.min(), sbp_true.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('True SBP (mmHg)', fontsize=12)
            axes[0, 0].set_ylabel('Predicted SBP (mmHg)', fontsize=12)
            axes[0, 0].set_title(f'SBP: MAE={sbp_mae:.2f}, r={sbp_r:.3f}', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3)
            
            # DBP scatter plot
            axes[0, 1].scatter(dbp_true, dbp_pred, alpha=0.5, s=20)
            axes[0, 1].plot([dbp_true.min(), dbp_true.max()], 
                           [dbp_true.min(), dbp_true.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('True DBP (mmHg)', fontsize=12)
            axes[0, 1].set_ylabel('Predicted DBP (mmHg)', fontsize=12)
            axes[0, 1].set_title(f'DBP: MAE={dbp_mae:.2f}, r={dbp_r:.3f}', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Bland-Altman plot for SBP
            sbp_mean = (sbp_true + sbp_pred) / 2
            sbp_diff = sbp_true - sbp_pred
            axes[1, 0].scatter(sbp_mean, sbp_diff, alpha=0.5, s=20)
            axes[1, 0].axhline(y=0, color='r', linestyle='-', lw=2)
            axes[1, 0].axhline(y=1.96*sbp_std, color='r', linestyle='--', lw=1)
            axes[1, 0].axhline(y=-1.96*sbp_std, color='r', linestyle='--', lw=1)
            axes[1, 0].set_xlabel('Mean SBP (mmHg)', fontsize=12)
            axes[1, 0].set_ylabel('Difference (mmHg)', fontsize=12)
            axes[1, 0].set_title('Bland-Altman Plot (SBP)', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Bland-Altman plot for DBP
            dbp_mean = (dbp_true + dbp_pred) / 2
            dbp_diff = dbp_true - dbp_pred
            axes[1, 1].scatter(dbp_mean, dbp_diff, alpha=0.5, s=20)
            axes[1, 1].axhline(y=0, color='r', linestyle='-', lw=2)
            axes[1, 1].axhline(y=1.96*dbp_std, color='r', linestyle='--', lw=1)
            axes[1, 1].axhline(y=-1.96*dbp_std, color='r', linestyle='--', lw=1)
            axes[1, 1].set_xlabel('Mean DBP (mmHg)', fontsize=12)
            axes[1, 1].set_ylabel('Difference (mmHg)', fontsize=12)
            axes[1, 1].set_title('Bland-Altman Plot (DBP)', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('bp_prediction_plots.png', dpi=300, bbox_inches='tight')
            print("\nPlots saved to 'bp_prediction_plots.png'")
            plt.show()
        
        return {
            'sbp_mae': sbp_mae,
            'sbp_rmse': sbp_rmse,
            'sbp_std': sbp_std,
            'sbp_r2': sbp_r2,
            'sbp_r': sbp_r,
            'sbp_mape': sbp_mape,
            'sbp_mean_error_percent': sbp_mean_error_percent,
            'sbp_std_error_percent': sbp_std_error_percent,
            'sbp_aami_pass': sbp_aami,
            'sbp_bhs_grade': sbp_bhs,
            'dbp_mae': dbp_mae,
            'dbp_rmse': dbp_rmse,
            'dbp_std': dbp_std,
            'dbp_r2': dbp_r2,
            'dbp_r': dbp_r,
            'dbp_mape': dbp_mape,
            'dbp_mean_error_percent': dbp_mean_error_percent,
            'dbp_std_error_percent': dbp_std_error_percent,
            'dbp_aami_pass': dbp_aami,
            'dbp_bhs_grade': dbp_bhs
        }
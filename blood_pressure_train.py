import pandas as pd
import numpy as np
import os
from itertools import product
import time
from sklearn.preprocessing import MinMaxScaler # or StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_data(path="datasets/CNAP_blood_pressure.csv"):
    """Load and preprocess data"""
    # load data
    df = pd.read_csv(path)
    
    # x y split
    feature_cols = df.columns[:300]
    target_cols = ['SBP', 'DBP']
    groups = df['subject'].values
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # apply MinMaxScaler
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    # reshape to (samples, timesteps, features)
    X = X.reshape((-1, 3, 100))
    X = np.transpose(X, (0, 2, 1))
    
    return X, y, scaler_y, groups

def build_model(config):
    """Build the neural network model based on configuration"""
    model = Sequential()
    model.add(Input(shape=(100, 3)))
    
    if config['ARCHITECTURE'] == 'pure_recurrent':
        # Pure recurrent architecture: all hidden layers are LSTM/GRU
        recurrent_layers = config['RECURRENT_LAYERS']
        
        if config['CLASSIFICADOR'] == 'LSTM':
            for idx, units in enumerate(recurrent_layers):
                if idx < len(recurrent_layers) - 1:
                    model.add(LSTM(units=units, activation=config['ACTIVATION'], return_sequences=True))
                else:
                    model.add(LSTM(units=units, activation=config['ACTIVATION'], return_sequences=False))
        elif config['CLASSIFICADOR'] == 'GRU':
            for idx, units in enumerate(recurrent_layers):
                if idx < len(recurrent_layers) - 1:
                    model.add(GRU(units=units, activation=config['ACTIVATION'], return_sequences=True))
                else:
                    model.add(GRU(units=units, activation=config['ACTIVATION'], return_sequences=False))
    
    elif config['ARCHITECTURE'] == 'mixed':
        # Mixed architecture: recurrent layers + dense layers
        recurrent_layers = config['RECURRENT_LAYERS']
        dense_layers = config['DENSE_LAYERS']
        
        # Add recurrent layers
        if config['CLASSIFICADOR'] == 'LSTM':
            for idx, units in enumerate(recurrent_layers):
                if idx < len(recurrent_layers) - 1:
                    model.add(LSTM(units=units, activation=config['ACTIVATION'], return_sequences=True))
                else:
                    model.add(LSTM(units=units, activation=config['ACTIVATION'], return_sequences=False))
        elif config['CLASSIFICADOR'] == 'GRU':
            for idx, units in enumerate(recurrent_layers):
                if idx < len(recurrent_layers) - 1:
                    model.add(GRU(units=units, activation=config['ACTIVATION'], return_sequences=True))
                else:
                    model.add(GRU(units=units, activation=config['ACTIVATION'], return_sequences=False))
        
        # Add dense layers
        for units in dense_layers:
            model.add(Dense(units=units, activation='relu'))
    
    # Output layer for regression (SBP, DBP)
    model.add(Dense(2, activation='linear'))
    
    # Choose optimizer with learning rate
    if config['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=config['LEARNING_RATE'])
    elif config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=config['LEARNING_RATE'])
    
    # Compile model
    model.compile(
        optimizer=optimizer, 
        loss='mse', 
        metrics=['mae', tf.keras.metrics.MeanAbsolutePercentageError()]
    )
    
    return model

def train_model(config):
    """Train and evaluate model using Leave-One-Group-Out cross-validation"""
    # define file paths
    path = "datasets/CNAP_blood_pressure.csv"
    
    # load data
    X, y, scaler_y, groups = load_data(path)
    
    # Leave-One-Group-Out cross-validation
    logo = LeaveOneGroupOut()
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        print(f"Processing fold {fold + 1}")
        test_subject = groups[test_idx][0]
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]
        
        print(f'Train shape: {X_train.shape}, Test shape: {X_val.shape}')
        
        # Build model using config
        model = build_model(config)
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['EPOCHS'],
            batch_size=config['BATCH_SIZE'],
            callbacks=[early_stop],
            verbose=0  # Silent training for grid search
        )

        # Rescale predictions back to original scale
        y_pred = model.predict(X_val)
        y_pred_rescaled = scaler_y.inverse_transform(y_pred)
        y_val_rescaled = scaler_y.inverse_transform(y_val)

        # Evaluate on the original scale
        mae = mean_absolute_error(y_val_rescaled, y_pred_rescaled)
        mape = mean_absolute_percentage_error(y_val_rescaled, y_pred_rescaled)
        loss = mean_squared_error(y_val_rescaled, y_pred_rescaled)

        # Store results
        results.append({
            'fold': fold + 1,
            'subject': test_subject,
            'loss': loss,
            'mae': mae,
            'mape': mape
        })
        
        # Clear memory
        del model
        tf.keras.backend.clear_session()
    
    # Calculate metrics across all folds
    results_df = pd.DataFrame(results)
    metrics = {
        'mean_loss': results_df['loss'].mean(),
        'std_loss': results_df['loss'].std(),
        'mean_mae': results_df['mae'].mean(),
        'std_mae': results_df['mae'].std(),
        'mean_mape': results_df['mape'].mean(),
        'std_mape': results_df['mape'].std()
    }
    
    return metrics, results_df

def grid_search(param_grid):
    """Perform grid search over all parameter combinations"""
    path_grid_search = os.path.join('grid_search/')
    
    # Create directory for grid search results
    if not os.path.exists(path_grid_search):
        os.makedirs(path_grid_search)
    
    # Get the keys and values for the grid search
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    # Generate all combinations of parameters
    combinations = list(product(*values))
    
    # Transform combinations into a list of dictionaries
    complete_grid = [dict(zip(keys, parameters)) for parameters in combinations]
    
    print(f"Total combinations to evaluate: {len(complete_grid)}")
    
    all_results = []
    all_fold_results = []
    
    for i, config in enumerate(complete_grid):
        print(f"\nEvaluating combination {i + 1}/{len(complete_grid)}")
        print(f"Config: {config}")
        
        start_time = time.time()
        
        try:
            # Train and evaluate model
            metrics, fold_results = train_model(config)
            
            # Add configuration info to fold results for detailed CSV
            fold_results_with_config = fold_results.copy()
            
            # FIX: Properly add configuration columns to each row
            for key, value in config.items():
                # Convert tuple/list values to string for CSV compatibility
                if isinstance(value, (tuple, list)):
                    fold_results_with_config[key] = str(value)
                else:
                    fold_results_with_config[key] = value
            
            fold_results_with_config['combination_id'] = i + 1
            fold_results_with_config['execution_time'] = time.time() - start_time
            
            all_fold_results.append(fold_results_with_config)
            
            # Create summary result row with means and stds
            result_row = {
                'combination_id': i + 1,
                'CLASSIFICADOR': config['CLASSIFICADOR'],
                'ARCHITECTURE': config['ARCHITECTURE'],
                'RECURRENT_LAYERS': str(config['RECURRENT_LAYERS']),
                'DENSE_LAYERS': str(config['DENSE_LAYERS']),
                'BATCH_SIZE': config['BATCH_SIZE'],
                'LEARNING_RATE': config['LEARNING_RATE'],
                'EPOCHS': config['EPOCHS'],
                'ACTIVATION': config['ACTIVATION'],
                'OPTIMIZER': config['OPTIMIZER'],
                'mean_loss': metrics['mean_loss'],
                'std_loss': metrics['std_loss'],
                'mean_mae': metrics['mean_mae'],
                'std_mae': metrics['std_mae'],
                'mean_mape': metrics['mean_mape'],
                'std_mape': metrics['std_mape'],
                'execution_time': time.time() - start_time
            }
            
            all_results.append(result_row)
            
            print(f"Results: Loss={metrics['mean_loss']:.4f}±{metrics['std_loss']:.4f}, "
                  f"MAE={metrics['mean_mae']:.4f}±{metrics['std_mae']:.4f}, "
                  f"MAPE={metrics['mean_mape']:.4f}±{metrics['std_mape']:.4f}")
            print(f"Execution time: {result_row['execution_time']:.2f} seconds")
            
        except Exception as e:
            print(f"Error in combination {i + 1}: {str(e)}")
            # Add failed combination with error info
            error_row = {
                'combination_id': i + 1,
                'CLASSIFICADOR': config['CLASSIFICADOR'],
                'ARCHITECTURE': config.get('ARCHITECTURE', 'N/A'),  # Use .get() for safety
                'RECURRENT_LAYERS': str(config['RECURRENT_LAYERS']),
                'DENSE_LAYERS': str(config['DENSE_LAYERS']),
                'BATCH_SIZE': config['BATCH_SIZE'],
                'LEARNING_RATE': config['LEARNING_RATE'],
                'EPOCHS': config['EPOCHS'],
                'ACTIVATION': config['ACTIVATION'],
                'OPTIMIZER': config['OPTIMIZER'],
                'mean_loss': np.nan,
                'std_loss': np.nan,
                'mean_mae': np.nan,
                'std_mae': np.nan,
                'mean_mape': np.nan,
                'std_mape': np.nan,
                'execution_time': time.time() - start_time,
                'error': str(e)
            }
            all_results.append(error_row)
    
    # Save complete summary results (main CSV with means and stds)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(path_grid_search, 'grid_search_summary_results.csv'), index=False)
    
    # Save detailed fold-by-fold results
    if all_fold_results:
        detailed_df = pd.concat(all_fold_results, ignore_index=True)
        detailed_df.to_csv(os.path.join(path_grid_search, 'grid_search_detailed_results.csv'), index=False)
    
    # Find best configuration
    valid_results = results_df.dropna(subset=['mean_loss'])
    if not valid_results.empty:
        best_idx = valid_results['mean_loss'].idxmin()
        best_config = valid_results.iloc[best_idx]
        
        print(f"\n{'='*50}")
        print("GRID SEARCH COMPLETE")
        print(f"{'='*50}")
        print(f"Total combinations evaluated: {len(complete_grid)}")
        print(f"Successful combinations: {len(valid_results)}")
        print(f"Failed combinations: {len(complete_grid) - len(valid_results)}")
        print(f"\nBest configuration (Combination ID: {best_config['combination_id']}):")
        for key in param_grid.keys():
            print(f"  {key}: {best_config[key]}")
        print(f"Best results:")
        print(f"  Loss: {best_config['mean_loss']:.4f} ± {best_config['std_loss']:.4f}")
        print(f"  MAE: {best_config['mean_mae']:.4f} ± {best_config['std_mae']:.4f}")
        print(f"  MAPE: {best_config['mean_mape']:.4f} ± {best_config['std_mape']:.4f}")
        
        # Save best configuration to separate file
        best_config_df = pd.DataFrame([best_config])
        best_config_df.to_csv(os.path.join(path_grid_search, 'best_configuration.csv'), index=False)
        
        print(f"\nResults saved to:")
        print(f"  - Summary results: grid_search/grid_search_summary_results.csv")
        print(f"  - Detailed results: grid_search/grid_search_detailed_results.csv")
        print(f"  - Best configuration: grid_search/best_configuration.csv")
        
        return results_df, best_config
    else:
        print("No successful combinations found!")
        return results_df, None

if __name__ == "__main__":
    # Small test grid for quick validation
    ### ADD DROUPOUT LAYER OPTION???
    test_param_grid = {
        'CLASSIFICADOR': ['LSTM', 'GRU'],
        'ARCHITECTURE': ['mixed'],
        'RECURRENT_LAYERS': [(64, 32)],
        'DENSE_LAYERS': [(32,)],
        'BATCH_SIZE': [16, 32],
        'LEARNING_RATE': [0.001, 0.01],
        'EPOCHS': [25, 50],
        'ACTIVATION': ['relu', 'tanh'],
        'OPTIMIZER': ['adam']
    }

    # Define layer configurations for different architectures
    # RECURRENT_LAYERS = [
    #     (32,), (64,), (128,),
    #     (8, 4), (16, 8), (32, 16), (64, 32), (128, 64),
    #     (32, 16, 8), (64, 32, 16)
    # ]

    # DENSE_LAYERS = [
    #     (32,), (64,), (128,), (256,),
    #     (64, 32), (128, 64), (256, 128),
    #     (128, 64, 32), (256, 128, 64)
    # ]

    # param_grid = {
    #     'CLASSIFICADOR': ['LSTM', 'GRU'],
    #     'ARCHITECTURE': ['pure_recurrent', 'mixed'],  # architecture type
    #     'RECURRENT_LAYERS': RECURRENT_LAYERS,
    #     'DENSE_LAYERS': DENSE_LAYERS,
    #     'BATCH_SIZE': [8, 16, 32, 64, 128],
    #     'LEARNING_RATE': [0.001, 0.01, 0.1],
    #     'EPOCHS': [25, 50, 100, 200, 400],
    #     'ACTIVATION': ['relu', 'tanh'],
    #     'OPTIMIZER': ['adam', 'sgd']
    # }

    results_df, best_config = grid_search(test_param_grid)
    
    # Or run the full grid search
    # results_df, best_config = grid_search(param_grid)
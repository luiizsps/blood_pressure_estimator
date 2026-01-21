import pandas as pd
import numpy as np
import os
from itertools import product
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    
    return X, y, scaler_y, groups

def build_model(config):
    """Build the Dense neural network model based on configuration"""
    model = Sequential()
    
    # Input layer with 300 features (flattened from original 3x100)
    model.add(Input(shape=(300,)))
    
    # Add Dense layers based on configuration
    LAYERS = config['LAYERS']
    
    for idx, units in enumerate(LAYERS):
        # Add Dense layer
        model.add(Dense(filters=units, activation=config['ACTIVATION'], kernel_size=5))
    
    # Output layer for regression (SBP, DBP)
    model.add(Dense(2, activation='linear'))
    
    # Choose optimizer with learning rate
    if config['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=config['LEARNING_RATE'])
    elif config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=config['LEARNING_RATE'])
    elif config['OPTIMIZER'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['LEARNING_RATE'])
    
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
        #model = build_model(config)

        base_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        # Wrap for multioutput (SBP + DBP)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.get('PATIENCE', 10),
            restore_best_weights=True
        )
        
        # Train model
        # history = model.fit(
        #     X_train, y_train,
        #     validation_data=(X_val, y_val),
        #     epochs=config['EPOCHS'],
        #     batch_size=config['BATCH_SIZE'],
        #     callbacks=[early_stop],
        #     verbose=0 
        # )

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
    path_grid_search = os.path.join('grid_search_dense/')
    
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
            
            # Properly add configuration columns to each row
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
                'LAYERS': str(config['LAYERS']),
                'BATCH_SIZE': config['BATCH_SIZE'],
                'LEARNING_RATE': config['LEARNING_RATE'],
                'EPOCHS': config['EPOCHS'],
                'ACTIVATION': config['ACTIVATION'],
                'OPTIMIZER': config['OPTIMIZER'],
                'DROPOUT_RATE': config.get('DROPOUT_RATE', 0),
                'USE_BATCH_NORM': config.get('USE_BATCH_NORM', False),
                'USE_LR_REDUCTION': config.get('USE_LR_REDUCTION', False),
                'PATIENCE': config.get('PATIENCE', 10),
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
                'LAYERS': str(config['LAYERS']),
                'BATCH_SIZE': config['BATCH_SIZE'],
                'LEARNING_RATE': config['LEARNING_RATE'],
                'EPOCHS': config['EPOCHS'],
                'ACTIVATION': config['ACTIVATION'],
                'OPTIMIZER': config['OPTIMIZER'],
                'DROPOUT_RATE': config.get('DROPOUT_RATE', 0),
                'USE_BATCH_NORM': config.get('USE_BATCH_NORM', False),
                'USE_LR_REDUCTION': config.get('USE_LR_REDUCTION', False),
                'PATIENCE': config.get('PATIENCE', 10),
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
        for key in ['LAYERS', 'BATCH_SIZE', 'LEARNING_RATE', 'EPOCHS', 
                   'ACTIVATION', 'OPTIMIZER', 'DROPOUT_RATE', 'USE_BATCH_NORM']:
            if key in best_config:
                print(f"  {key}: {best_config[key]}")
        print(f"Best results:")
        print(f"  Loss: {best_config['mean_loss']:.4f} ± {best_config['std_loss']:.4f}")
        print(f"  MAE: {best_config['mean_mae']:.4f} ± {best_config['std_mae']:.4f}")
        print(f"  MAPE: {best_config['mean_mape']:.4f} ± {best_config['std_mape']:.4f}")
        
        # Save best configuration to separate file
        best_config_df = pd.DataFrame([best_config])
        best_config_df.to_csv(os.path.join(path_grid_search, 'best_configuration.csv'), index=False)
        
        print(f"\nResults saved to:")
        print(f"  - Summary results: {path_grid_search}grid_search_summary_results.csv")
        print(f"  - Detailed results: {path_grid_search}grid_search_detailed_results.csv")
        print(f"  - Best configuration: {path_grid_search}best_configuration.csv")
        
        return results_df, best_config
    else:
        print("No successful combinations found!")
        return results_df, None

def train_xgboost(config):
    """Treina e avalia o modelo XGBoost usando Leave-One-Group-Out CV"""
    X, y, scaler_y, groups = load_data("datasets/CNAP_blood_pressure.csv")
    logo = LeaveOneGroupOut()
    results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        print(f"\nProcessing fold {fold + 1}")
        test_subject = groups[test_idx][0]

        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]

        # Modelo base com parâmetros vindos do grid
        base_model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            learning_rate=config['learning_rate'],
            max_depth=config['max_depth'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            gamma=config.get('gamma', 0),
            min_child_weight=config.get('min_child_weight', 1),
            random_state=42,
            n_jobs=-1,
        )

        # Multi-output para prever SBP e DBP
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # Reescalar previsões
        y_pred_rescaled = scaler_y.inverse_transform(y_pred)
        y_val_rescaled = scaler_y.inverse_transform(y_val)

        # Métricas
        mae = mean_absolute_error(y_val_rescaled, y_pred_rescaled)
        mape = mean_absolute_percentage_error(y_val_rescaled, y_pred_rescaled)
        mse = mean_squared_error(y_val_rescaled, y_pred_rescaled)

        results.append({
            'fold': fold + 1,
            'subject': test_subject,
            'loss': mse,
            'mae': mae,
            'mape': mape
        })

        del model
        tf.keras.backend.clear_session()

    # Média das métricas
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


def grid_search_xgb(param_grid):
    """Executa grid search no XGBoost"""
    path_grid_search = "grid_search_xgb/"
    os.makedirs(path_grid_search, exist_ok=True)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    print(f"Total de combinações: {len(combinations)}\n")

    all_results = []

    for i, config in enumerate(combinations):
        print(f"Treinando combinação {i + 1}/{len(combinations)}")
        print(config)

        start_time = time.time()
        try:
            metrics, _ = train_xgboost(config)
            result = {**config,
                      'mean_loss': metrics['mean_loss'],
                      'mean_mae': metrics['mean_mae'],
                      'mean_mape': metrics['mean_mape'],
                      'execution_time': time.time() - start_time}
            all_results.append(result)
        except Exception as e:
            print(f"Erro: {e}")
            result = {**config,
                      'mean_loss': np.nan,
                      'mean_mae': np.nan,
                      'mean_mape': np.nan,
                      'error': str(e)}
            all_results.append(result)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(path_grid_search, 'xgb_grid_search_results.csv'), index=False)

    best = df_results.loc[df_results['mean_loss'].idxmin()]
    print("\nMelhor configuração encontrada:")
    print(best)
    return df_results, best


if __name__ == "__main__":
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'min_child_weight': [1, 3, 5]
    }

    print("Executando grid search no XGBoost...")
    results_df, best_config = grid_search_xgb(param_grid_xgb)

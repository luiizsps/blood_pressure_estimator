import pandas as pd
import numpy as np
import os
from itertools import product
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Input, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import random

# -----------------------
# CONFIGURAÇÃO GLOBAL
# -----------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# -----------------------
# FUNÇÃO DE CARREGAMENTO DE DADOS
# -----------------------
def load_data(path="datasets/CNAP_blood_pressure.csv"):
    """Load and preprocess data"""
    df = pd.read_csv(path)
    
    feature_cols = df.columns[:300]
    target_cols = ['SBP', 'DBP']
    groups = df['subject'].values
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Normalização
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    # Reshape para formato (samples, timesteps, features)
    X = X.reshape((-1, 3, 100))
    X = np.transpose(X, (0, 2, 1))
    
    return X, y, scaler_y, groups

# -----------------------
# CONSTRUÇÃO DO MODELO CNN1D
# -----------------------
def build_model(config):
    """Build CNN1D model for regression"""
    model = Sequential()
    model.add(Input(shape=(100, 3)))
    
    # Camadas convolucionais
    for idx, (filters, kernel_size) in enumerate(zip(config['FILTERS'], config['KERNEL_SIZES'])):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=config['ACTIVATION']))
        if config.get('USE_POOLING', False):
            model.add(MaxPooling1D(pool_size=2))
        if config.get('DROPOUT_RATE', 0) > 0:
            model.add(Dropout(config['DROPOUT_RATE']))
    
    # Flatten antes das densas
    model.add(Flatten())
    
    # Camadas densas
    for units in config['DENSE_LAYERS']:
        model.add(Dense(units=units, activation=config['ACTIVATION']))
        if config.get('DROPOUT_RATE', 0) > 0:
            model.add(Dropout(config['DROPOUT_RATE']))
    
    # Saída (2 neurônios para SBP e DBP)
    model.add(Dense(2, activation='linear'))
    
    # Otimizador
    if config['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=config['LEARNING_RATE'])
    elif config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=config['LEARNING_RATE'])
    
    # Compilação
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.MeanAbsolutePercentageError()]
    )
    
    return model

# -----------------------
# TREINAMENTO E AVALIAÇÃO COM CROSS-VALIDATION
# -----------------------
def train_model(config):
    """Train and evaluate CNN1D model using Leave-One-Group-Out cross-validation"""
    path = "datasets/CNAP_blood_pressure.csv"
    X, y, scaler_y, groups = load_data(path)
    
    logo = LeaveOneGroupOut()
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        print(f"Processing fold {fold + 1}")
        test_subject = groups[test_idx][0]
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_val.shape}")
        
        model = build_model(config)
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Treinamento silencioso
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['EPOCHS'],
            batch_size=config['BATCH_SIZE'],
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predições e métricas
        y_pred = model.predict(X_val)
        y_pred_rescaled = scaler_y.inverse_transform(y_pred)
        y_val_rescaled = scaler_y.inverse_transform(y_val)
        
        mae = mean_absolute_error(y_val_rescaled, y_pred_rescaled)
        mape = mean_absolute_percentage_error(y_val_rescaled, y_pred_rescaled)
        loss = mean_squared_error(y_val_rescaled, y_pred_rescaled)
        
        results.append({
            'fold': fold + 1,
            'subject': test_subject,
            'loss': loss,
            'mae': mae,
            'mape': mape
        })
        
        del model
        tf.keras.backend.clear_session()
    
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

# -----------------------
# GRID SEARCH
# -----------------------
def grid_search(param_grid):
    """Perform grid search over CNN1D configurations"""
    path_grid_search = os.path.join('grid_search/')
    os.makedirs(path_grid_search, exist_ok=True)
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    complete_grid = [dict(zip(keys, parameters)) for parameters in combinations]
    
    print(f"Total combinations to evaluate: {len(complete_grid)}")
    
    all_results = []
    all_fold_results = []
    
    for i, config in enumerate(complete_grid):
        print(f"\nEvaluating combination {i + 1}/{len(complete_grid)}")
        print(f"Config: {config}")
        
        start_time = time.time()
        
        try:
            metrics, fold_results = train_model(config)
            
            fold_results_with_config = fold_results.copy()
            for key, value in config.items():
                fold_results_with_config[key] = str(value) if isinstance(value, (tuple, list)) else value
            fold_results_with_config['combination_id'] = i + 1
            fold_results_with_config['execution_time'] = time.time() - start_time
            
            all_fold_results.append(fold_results_with_config)
            
            result_row = {
                'combination_id': i + 1,
                'CLASSIFICADOR': config['CLASSIFICADOR'],
                'FILTERS': str(config['FILTERS']),
                'KERNEL_SIZES': str(config['KERNEL_SIZES']),
                'DENSE_LAYERS': str(config['DENSE_LAYERS']),
                'BATCH_SIZE': config['BATCH_SIZE'],
                'LEARNING_RATE': config['LEARNING_RATE'],
                'EPOCHS': config['EPOCHS'],
                'ACTIVATION': config['ACTIVATION'],
                'OPTIMIZER': config['OPTIMIZER'],
                'DROPOUT_RATE': config['DROPOUT_RATE'],
                'USE_POOLING': config['USE_POOLING'],
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
            
        except Exception as e:
            print(f"Error in combination {i + 1}: {str(e)}")
            error_row = {k: config.get(k, 'N/A') for k in keys}
            error_row.update({
                'combination_id': i + 1,
                'mean_loss': np.nan,
                'std_loss': np.nan,
                'mean_mae': np.nan,
                'std_mae': np.nan,
                'mean_mape': np.nan,
                'std_mape': np.nan,
                'execution_time': time.time() - start_time,
                'error': str(e)
            })
            all_results.append(error_row)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(path_grid_search, 'grid_search_summary_results.csv'), index=False)
    
    if all_fold_results:
        detailed_df = pd.concat(all_fold_results, ignore_index=True)
        detailed_df.to_csv(os.path.join(path_grid_search, 'grid_search_detailed_results.csv'), index=False)
    
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
        print(f"\nBest results:")
        print(f"  Loss: {best_config['mean_loss']:.4f} ± {best_config['std_loss']:.4f}")
        print(f"  MAE: {best_config['mean_mae']:.4f} ± {best_config['std_mae']:.4f}")
        print(f"  MAPE: {best_config['mean_mape']:.4f} ± {best_config['std_mape']:.4f}")
        
        best_config_df = pd.DataFrame([best_config])
        best_config_df.to_csv(os.path.join(path_grid_search, 'best_configuration.csv'), index=False)
        
        return results_df, best_config
    else:
        print("No successful combinations found!")
        return results_df, None

# -----------------------
# EXECUÇÃO DO GRID SEARCH CNN1D
# -----------------------
if __name__ == "__main__":
    cnn_param_grid = {
        'CLASSIFICADOR': ['CNN1D'],
        'FILTERS': [(64, 32)],
        'KERNEL_SIZES': [(5,), (5, 3)],
        'DENSE_LAYERS': [(64,)],
        'BATCH_SIZE': [16, 32],
        'LEARNING_RATE': [0.001, 0.01],
        'EPOCHS': [50],
        'ACTIVATION': ['relu'],
        'OPTIMIZER': ['adam'],
        'DROPOUT_RATE': [0.0, 0.5],
        'USE_POOLING': [True, False]
    }

    results_df, best_config = grid_search(cnn_param_grid)

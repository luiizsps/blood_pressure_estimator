import os
import csv
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def train_xgboost_and_evaluate(x_train, y_train, params, x_test=None, y_test=None, dump=None):
    """
    Treina e avalia um modelo XGBoost para múltiplos targets.
    
    Args:
        x_train: Features de treino
        y_train: Targets de treino (pode ser multidimensional)
        params: Parâmetros do XGBoost
        x_test: Features de teste (opcional)
        y_test: Targets de teste (opcional)
        dump: Path para salvar modelos (opcional)
    
    Returns:
        models, mapes, rmses, maes, preds
    """
    models = []
    preds = []
    rmses, mapes, maes = [], [], []

    # Garantir que y_train é 2D
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test is not None and len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)

    for i in range(y_train.shape[1]):
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, y_train[:, i])
        models.append(model)

        if x_test is not None and y_test is not None:
            pred = model.predict(x_test)
            preds.append(pred)
            rmses.append(np.sqrt(mean_squared_error(y_test[:, i], pred)))
            mapes.append(mean_absolute_percentage_error(y_test[:, i], pred))
            maes.append(mean_absolute_error(y_test[:, i], pred))
        else:
            preds.append(None)
            rmses.append(None)
            mapes.append(None)
            maes.append(None)

    if dump:
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        for i, model in enumerate(models):
            model.save_model(f"{dump}_target{i}.json")

    return models, mapes, rmses, maes, preds


def run_offline_study(target, validation_mode, params, Z_within_subj, dataset_input_path, model_path):
    """
    Realiza estudo offline com XGBoost.
    
    Modos de validação:
    - 'LOPO': Leave-One-Participant-Out (treina em N-1, testa em 1)
    - 'within_subject': Train/test split dentro de cada participante
    """
    # Configuração por tipo de target
    if target.lower() == "bp":
        subjs = ["MJ", "SB", "EW", "PV", "JR", "HS", "JB", "SK"]
        datafile_prefix = f"CNAP_Z_{Z_within_subj}" if Z_within_subj else "CNAP"
        y_cols = [300, 301]
    elif target.lower() == "sv":
        subjs = ["HH", "HS", "HYS", "JB", "PT", "PV", "SK", "SKR", "SS", "TH", "ZL"]
        datafile_prefix = f"NICOM_Z_{Z_within_subj}" if Z_within_subj else "NICOM"
        y_cols = [300]
    else:
        raise ValueError(f"Target '{target}' não reconhecido. Use 'bp' ou 'sv'.")

    subj_rmses, subj_mapes, subj_maes = [], [], []
    altman_subj_id, altman_gold, altman_test = [], [], []

    if validation_mode.lower() == 'lopo':
        # Leave-One-Participant-Out
        print("🔄 Modo LOPO: Treinando com N-1 participantes, testando em 1")
        
        for test_subj in subjs:
            print(f"\n{'='*60}")
            print(f"📊 Testando em: {test_subj}")
            print(f"🎯 Treinando em: {[s for s in subjs if s != test_subj]}")
            print('='*60)
            
            # Carregar dados de treino (todos exceto test_subj)
            X_train_list, y_train_list = [], []
            for train_subj in subjs:
                if train_subj == test_subj:
                    continue
                
                filename = os.path.join(dataset_input_path, f'{datafile_prefix}_{train_subj}.csv')
                if not os.path.exists(filename):
                    print(f"⚠️  Arquivo não encontrado: {filename}")
                    continue
                    
                df = pd.read_csv(filename)
                X_train_list.append(df.iloc[:, :300].values)
                y_train_list.append(df.iloc[:, y_cols].values)
            
            # Concatenar todos os dados de treino
            X_train = np.vstack(X_train_list)
            y_train = np.vstack(y_train_list)
            
            # Carregar dados de teste
            test_filename = os.path.join(dataset_input_path, f'{datafile_prefix}_{test_subj}.csv')
            if not os.path.exists(test_filename):
                print(f"⚠️  Arquivo de teste não encontrado: {test_filename}")
                continue
                
            df_test = pd.read_csv(test_filename)
            X_test = df_test.iloc[:, :300].values
            y_test = df_test.iloc[:, y_cols].values
            
            # Treinar e avaliar
            models, mapes, rmses, maes, preds = train_xgboost_and_evaluate(
                X_train, y_train, params, X_test, y_test, dump=None
            )
            
            subj_rmses.append(rmses)
            subj_mapes.append(mapes)
            subj_maes.append(maes)
            
            # Armazenar predições para análise Bland-Altman
            altman_gold += list(y_test[:, 0])
            altman_test += list(preds[0])
            altman_subj_id += [f"P{subjs.index(test_subj)+1}"] * len(preds[0])
            
            for i, (m, r, a) in enumerate(zip(mapes, rmses, maes)):
                print(f"  Target {i}: MAPE={m:.4f}, RMSE={r:.4f}, MAE={a:.4f}")
    
    else:  # within_subject
        print("🔄 Modo Within-Subject: Train/test split em cada participante")
        
        for subj in subjs:
            print(f"\n📊 Participante: {subj}")
            filename = os.path.join(dataset_input_path, f'{datafile_prefix}_{subj}.csv')
            
            if not os.path.exists(filename):
                print(f"⚠️  Arquivo não encontrado: {filename}")
                continue
            
            df = pd.read_csv(filename)
            X = df.iloc[:, :300].values
            ys = df.iloc[:, y_cols].values

            x_train, x_test, y_train, y_test = train_test_split(
                X, ys, test_size=0.2, random_state=42, shuffle=True
            )

            models, mapes, rmses, maes, preds = train_xgboost_and_evaluate(
                x_train, y_train, params, x_test, y_test, dump=None
            )

            subj_rmses.append(rmses)
            subj_mapes.append(mapes)
            subj_maes.append(maes)
            altman_gold += list(y_test[:, 0])
            altman_test += list(preds[0])
            altman_subj_id += [f"P{subjs.index(subj)+1}"] * len(preds[0])

            for i, (m, r, a) in enumerate(zip(mapes, rmses, maes)):
                print(f"  Target {i}: MAPE={m:.4f}, RMSE={r:.4f}, MAE={a:.4f}")

    # Calcular médias gerais
    print(f"\n{'='*60}")
    print("📈 RESULTADOS GERAIS:")
    print('='*60)
    avg_rmse = np.mean([r[0] for r in subj_rmses])
    avg_mape = np.mean([m[0] for m in subj_mapes])
    avg_mae = np.mean([a[0] for a in subj_maes])
    print(f"RMSE médio: {avg_rmse:.4f}")
    print(f"MAPE médio: {avg_mape:.4f}")
    print(f"MAE médio: {avg_mae:.4f}")

    # Salvar métricas
    os.makedirs(model_path, exist_ok=True)
    metrics_path = os.path.join(model_path, f"{validation_mode}_{Z_within_subj}_metrics.csv")
    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subj", "rmse", "mape", "mae"])
        for s, r, m, a in zip(subjs, subj_rmses, subj_mapes, subj_maes):
            # Corrigido: salvar valores escalares, não listas
            writer.writerow([s, r[0], m[0], a[0]])

    # Salvar predições
    preds_path = os.path.join(model_path, f"{validation_mode}_predictions.csv")
    with open(preds_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subj", "predicted", "actual"])
        for i in range(len(altman_subj_id)):
            writer.writerow([altman_subj_id[i], altman_test[i], altman_gold[i]])

    print(f"\n✅ Métricas salvas em: {metrics_path}")
    print(f"✅ Predições salvas em: {preds_path}")


if __name__ == "__main__":
    xgboost_parameters = {
        'max_depth': 2,
        'n_estimators': 20,
        'learning_rate': 0.1,
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    # Exemplo: validação LOPO (verdadeira)
    run_offline_study(
        target='bp',
        validation_mode='LOPO',  # Agora implementado corretamente
        params=xgboost_parameters,
        Z_within_subj='minmax',
        dataset_input_path='ML/subject_data/',
        model_path='bp_inference_suite'
    )
    
    # Ou use 'within_subject' para train/test split individual
    # run_offline_study(
    #     target='bp',
    #     validation_mode='within_subject',
    #     params=xgboost_parameters,
    #     Z_within_subj='minmax',
    #     dataset_input_path='ML/subject_data/',
    #     model_path='bp_inference_suite'
    # )
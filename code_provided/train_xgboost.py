import os
import csv
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def train_xgboost_multioutput_and_evaluate(x_train, y_train, params, x_test=None, y_test=None, dump=None):
    model = MultiOutputRegressor(xgb.XGBRegressor(**params))
    model.fit(x_train, y_train)

    if x_test is not None and y_test is not None:
        preds = model.predict(x_test)

        rmses, mapes, maes = [], [], []
        for i in range(y_train.shape[1]):
            rmse = np.sqrt(mean_squared_error(y_test[:, i], preds[:, i]))
            mape = mean_absolute_percentage_error(y_test[:, i], preds[:, i])
            mae = mean_absolute_error(y_test[:, i], preds[:, i])
            rmses.append(rmse)
            mapes.append(mape)
            maes.append(mae)
    else:
        preds, rmses, mapes, maes = None, None, None, None

    if dump:
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        model.estimators_[0].save_model(f"{dump}_SBP.json")
        if y_train.shape[1] > 1:
            model.estimators_[1].save_model(f"{dump}_DBP.json")

    return model, mapes, rmses, maes, preds


def run_offline_study(target, validation_mode, params, Z_within_subj, dataset_input_path, model_path):
    if target.lower() == "bp":
        subjs = ["MJ", "SB", "EW", "PV", "JR", "HS", "JB", "SK"]
        datafile_prefix = f"CNAP_Z_{Z_within_subj}" if Z_within_subj else "CNAP"
        y_cols = [300, 301]
        target_names = ["SBP", "DBP"]
    elif target.lower() == "sv":
        subjs = ["HH", "HS", "HYS", "JB", "PT", "PV", "SK", "SKR", "SS", "TH", "ZL"]
        datafile_prefix = f"NICOM_Z_{Z_within_subj}" if Z_within_subj else "NICOM"
        y_cols = [300]
        target_names = ["SV"]

    subj_rmses, subj_mapes, subj_maes = [], [], []
    all_preds, all_actuals, all_subjs = [], [], []

    for subj in subjs:
        print(f"Treinando participante {subj}...")
        filename = os.path.join(dataset_input_path, f'{datafile_prefix}_{subj}.csv')

        df = pd.read_csv(filename)

        X = np.array(df.iloc[:, :300])
        ys = np.array(df.iloc[:, y_cols])

        x_train, x_test, y_train, y_test = train_test_split(
            X, ys, test_size=0.2, random_state=42, shuffle=True
        )

        model, mapes, rmses, maes, preds = train_xgboost_multioutput_and_evaluate(
            x_train, y_train, params, x_test, y_test, dump=None
        )

        subj_rmses.append(rmses)
        subj_mapes.append(mapes)
        subj_maes.append(maes)

        # ✅ Armazenar predições e valores reais para este participante
        all_preds.append(preds)
        all_actuals.append(y_test)
        all_subjs += [subj] * len(preds)

        for i, (m, r, a) in enumerate(zip(mapes, rmses, maes)):
            print(f"Subj {subj} - {target_names[i]}: MAPE={m:.4f}, RMSE={r:.4f}, MAE={a:.4f}")

    # Salvar métricas
    os.makedirs(model_path, exist_ok=True)
    metrics_path = os.path.join(model_path, f"{validation_mode}_{Z_within_subj}_metrics.csv")

    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["subj"]
        for t in target_names:
            header += [f"{t}_RMSE", f"{t}_MAPE", f"{t}_MAE"]
        writer.writerow(header)

        for s, r, m, a in zip(subjs, subj_rmses, subj_mapes, subj_maes):
            row = [s]
            for i in range(len(target_names)):
                row += [r[i], m[i], a[i]]
            writer.writerow(row)

    # Juntar todas as predições
    all_preds = np.vstack(all_preds)
    all_actuals = np.vstack(all_actuals)

    preds_path = os.path.join(model_path, f"{validation_mode}_predictions.csv")
    with open(preds_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["subj"]
        for t in target_names:
            header += [f"{t}_pred", f"{t}_actual"]
        writer.writerow(header)

        for i in range(len(all_subjs)):
            row = [all_subjs[i]]
            for j in range(len(target_names)):
                row += [all_preds[i, j], all_actuals[i, j]]
            writer.writerow(row)

    print("Treinamento XGBoost MultiOutput concluído.")


if __name__ == "__main__":
    xgboost_parameters = {
        'max_depth': 2,
        'n_estimators': 20,
        'learning_rate': 0.1,
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    run_offline_study(
        target='bp',
        validation_mode='LOPO',
        params=xgboost_parameters,
        Z_within_subj='minmax',
        dataset_input_path='ML/subject_data/',
        model_path='bp_inference_suite'
    )

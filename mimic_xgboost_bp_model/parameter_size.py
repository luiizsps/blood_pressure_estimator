import pickle
import os
import xgboost as xgb

# -------------------------------
# Paths
# -------------------------------
pkl_path = "xgboost_bp_model.pkl"
json_path = "xgboost_bp_model.json"

# -------------------------------
# Load the PKL model
# -------------------------------
with open(pkl_path, "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Extract the individual XGBoost Booster from each model in MultiOutputRegressor
# -------------------------------
json_paths = []  # To store paths of each JSON file

for i, regressor in enumerate(model.estimators_):  # Assuming it's a MultiOutputRegressor
    # Get the Booster from the regressor
    booster = regressor.get_booster()
    
    # Define the JSON path for each model
    individual_json_path = f"{json_path}_model_{i}.json"
    
    # Save the booster model as a .json file
    booster.save_model(individual_json_path)
    
    # Add the path to our list
    json_paths.append(individual_json_path)

# -------------------------------
# Measure sizes
# -------------------------------
pkl_size_kib = os.path.getsize(pkl_path) / 1024
json_sizes_kib = [os.path.getsize(path) / 1024 for path in json_paths]

# Print the sizes
print(f"PKL model size  : {pkl_size_kib:.2f} KiB")
for i, json_size in enumerate(json_sizes_kib):
    print(f"JSON model {i} size: {json_size:.2f} KiB")


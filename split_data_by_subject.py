import pandas as pd

bp_data_path = "datasets/CNAP_blood_pressure.csv"
sv_data_path = "datasets/NICOM_stroke_volume.csv"
output_path = "code_provided/ML/subject_data/"

bp_data = pd.read_csv(bp_data_path)
sv_data = pd.read_csv(sv_data_path)

print(bp_data["subject"].unique())
print(sv_data["subject"].unique())

# bp
bp_subjects = ["MJ", "SB", "EW", "PV", "JR", "HS", "JB", "SK"]

for i, subject in enumerate(bp_subjects):
    csv_name = f"CNAP_Z_minmax_{subject}"
    subject_data = bp_data[bp_data["subject"] == i]
    subject_data.to_csv(f"{output_path}{csv_name}.csv", index=False)


# sv
sv_subjects = ["HH", "HS", "HYS", "JB", "PT", "PV", "SK", "SKR", "SS", "TH", "ZL"]

for i, subject in enumerate(sv_subjects):
    csv_name = f"NICOM_Z_minmax_{subject}"
    subject_data = sv_data[sv_data["subject"] == i]
    subject_data.to_csv(f"{output_path}{csv_name}.csv", index=False)


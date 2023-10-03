import pandas as pd
import numpy as np
import os
import yaml

# Clean up memory
# Remove all objects
for obj in list(globals()):
    if not obj.startswith("__"):
        del globals()[obj]

# Parameters
PARAM = {
    "experimento": "DR6210",
    "exp_input": "CA6110",
    "variables_intrames": True,
    "metodo": "deflacion",
    "home": "~/buckets/b1/"
}

OUTPUT = {}


# Define a function to save output as YAML
def save_output():
    with open("output.yml", "w") as output_file:
        yaml.dump(OUTPUT, output_file)

# Add your own manual variables here
def add_variables_intra_month(dataset):
    # Beginning of section where changes should be made with new variables
    dataset["ctrx_quarter_normalizado"] = dataset["ctrx_quarter"]
    dataset.loc[dataset["cliente_antiguedad"] == 1, "ctrx_quarter_normalizado"] = dataset["ctrx_quarter"] * 5
    dataset.loc[dataset["cliente_antiguedad"] == 2, "ctrx_quarter_normalizado"] = dataset["ctrx_quarter"] * 2
    dataset.loc[dataset["cliente_antiguedad"] == 3, "ctrx_quarter_normalizado"] = dataset["ctrx_quarter"] * 1.2

    dataset["mpayroll_sobre_edad"] = dataset["mpayroll"] / dataset["cliente_edad"]

    # Combine MasterCard and Visa
    dataset["vm_status01"] = np.maximum(dataset["Master_status"], dataset["Visa_status"])
    dataset["vm_status02"] = dataset["Master_status"] + dataset["Visa_status"]
    dataset["vm_status03"] = np.maximum(np.where(dataset["Master_status"].isna(), 10, dataset["Master_status"]),
                                        np.where(dataset["Visa_status"].isna(), 10, dataset["Visa_status"]))
    dataset["vm_status04"] = np.where(dataset["Master_status"].isna(), 10, dataset["Master_status"]) + \
                             np.where(dataset["Visa_status"].isna(), 10, dataset["Visa_status"])
    dataset["vm_status05"] = np.where(dataset["Master_status"].isna(), 10, dataset["Master_status"]) + \
                             100 * np.where(dataset["Visa_status"].isna(), 10, dataset["Visa_status"])
    dataset["vm_status06"] = np.where(dataset["Visa_status"].isna(),
                                       np.where(dataset["Master_status"].isna(), 10, dataset["Master_status"]),
                                       dataset["Visa_status"])
    dataset["mv_status07"] = np.where(dataset["Master_status"].isna(),
                                       np.where(dataset["Visa_status"].isna(), 10, dataset["Visa_status"]),
                                       dataset["Master_status"])

    # Combine monetary values for MasterCard and Visa
    monetary_cols = [col for col in dataset.columns if col.startswith("Master_") or col.startswith("Visa_")]
    for col in monetary_cols:
        dataset[f"vm_{col}"] = dataset[f"Master_{col}"] + dataset[f"Visa_{col}"]

    # Continue adding your own new variables here

    # Replace infinite values with NaN
    infinites = dataset.apply(lambda col: sum(np.isinf(col)))
    infinites_qty = sum(infinites)
    if infinites_qty > 0:
        print(f"ATTENTION: There are {infinites_qty} infinite values in your dataset. They will be set to NaN.")
        dataset = dataset.replace([np.inf, -np.inf], np.nan)

    # Replace NaN values with 0
    nans = dataset.apply(lambda col: sum(np.isnan(col)))
    nans_qty = sum(nans)
    if nans_qty > 0:
        print(f"ATTENTION: There are {nans_qty} NaN values (0/0) in your dataset. They will be arbitrarily set to 0.")
        print("If you disagree with this decision, modify the program as desired.")
        dataset = dataset.fillna(0)

    return dataset

# Drift correction functions
def drift_deflation(dataset, monetary_cols):
    vfoto_mes = [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201910, 201911, 201912,
                202001, 202002, 202003, 202004, 202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012,
                202101, 202102, 202103, 202104, 202105, 202106, 202107, 202108, 202109]

    vIPC = [1.9903030878, 1.9174403544, 1.8296186587, 1.7728862972, 1.7212488323, 1.6776304408, 1.6431248196,
            1.5814483345, 1.4947526791, 1.4484037589, 1.3913580777, 1.3404220402, 1.3154288912, 1.2921698342,
            1.2472681797, 1.2300475145, 1.2118694724, 1.1881073259, 1.1693969743, 1.1375456949, 1.1065619600,
            1.0681100000, 1.0370000000, 1.0000000000, 0.9680542110, 0.9344152616, 0.8882274350, 0.8532444140,
            0.8251880213, 0.8003763543, 0.7763107219, 0.7566381305, 0.7289384687]

    tb_IPC = pd.DataFrame({"foto_mes": vfoto_mes, "IPC": vIPC})

    for col in monetary_cols:
        dataset = dataset.merge(tb_IPC, on="foto_mes", how="left", suffixes=("", "_IPC"))
        dataset[col] = dataset[col] * dataset["IPC"]

    return dataset

def drift_rank_simple(dataset, drift_cols):
    for col in drift_cols:
        dataset[col + "_rank"] = (dataset.groupby("foto_mes")[col].rank(method="random") - 1) / (
                dataset.groupby("foto_mes")[col].transform("count") - 1)
        dataset.drop(columns=[col], inplace=True)

def drift_rank_cero_fijo(dataset, drift_cols):
    for col in drift_cols:
        dataset[col + "_rank"] = 0
        dataset.loc[dataset[col] > 0, col + "_rank"] = dataset[dataset[col] > 0].groupby("foto_mes")[col].rank(
            method="random") / dataset.groupby("foto_mes")[col].transform("count")
        dataset.loc[dataset[col] < 0, col + "_rank"] = -dataset[dataset[col] < 0].groupby("foto_mes")[col].rank(
            method="random") / dataset.groupby("foto_mes")[col].transform("count")
        dataset.drop(columns=[col], inplace=True)

def drift_standardize(dataset, drift_cols):
    for col in drift_cols:
        mean = dataset.groupby("foto_mes")[col].transform("mean")
        std = dataset.groupby("foto_mes")[col].transform("std")
        dataset[col + "_normal"] = (dataset[col] - mean) / std
        dataset.drop(columns=[col], inplace=True)

# Main program starts here
OUTPUT["PARAM"] = PARAM
OUTPUT["time"] = {"start": pd.Timestamp.now().strftime("%Y%m%d %H%M%S")}

os.chdir(PARAM["home"])

# Load the dataset
dataset_input = os.path.join("./exp/", PARAM["exp_input"], "dataset.csv.gz")
dataset = pd.read_csv(dataset_input)

# Create the experiment folder
exp_folder = os.path.join("./exp/", PARAM["experimento"])
os.makedirs(exp_folder, exist_ok=True)

# Set the working directory for the experiment
os.chdir(exp_folder)

save_output()

# Save parameters to a YAML file
with open("parametros.yml", "w") as param_file:
    yaml.dump(PARAM, param_file)

# Add manual variables if needed
if PARAM["variables_intrames"]:
    dataset = add_variables_intra_month(dataset)

# Sort by ranking
dataset.sort_values(by=["foto_mes", "numero_de_cliente"], inplace=True)

# Define monetary columns
monetary_columns = [col for col in dataset.columns if col.startswith(("m", "Visa_m", "Master_m", "vm_m"))]

# Apply the data drifting correction method
if PARAM["metodo"] == "ninguno":
    print("No data drifting correction is applied")
elif PARAM["metodo"] == "rank_simple":
    drift_rank_simple(dataset, monetary_columns)
elif PARAM["metodo"] == "rank_cero_fijo":
    drift_rank_cero_fijo(dataset, monetary_columns)
elif PARAM["metodo"] == "deflacion":
    dataset = drift_deflation(dataset, monetary_columns)
elif PARAM["metodo"] == "estandarizar":
    drift_standardize(dataset, monetary_columns)

# Save the dataset
dataset.to_csv("dataset.csv.gz", index=False, compression="gzip")

# Save dataset columns information
tb_campos = pd.DataFrame({
    "pos": range(1, len(dataset.columns) + 1),
    "campo": dataset.columns,
    "tipo": dataset.dtypes.values,
    "nulos": dataset.isnull().sum().values,
    "ceros": (dataset == 0).sum().values
})

tb_campos.to_csv("dataset.campos.txt", sep="\t", index=False)

# Record dataset information
OUTPUT["dataset"] = {
    "ncol": len(dataset.columns),
    "nrow": len(dataset)
}

OUTPUT["time"]["end"] = pd.Timestamp.now().strftime("%Y%m%d %H%M%S")

# Save the final timestamp
with open("zRend.txt", "a") as file:
    file.write(OUTPUT["time"]["end"])

import os
import pandas as pd
import numpy as np
import random
import gc
import yaml
from data.table import fread, fwrite

# Parameters
PARAM = {
    "experimento": "TS6410",
    "exp_input": "FE6310",
    "future": [202109],
    "final_train": [202107, 202106, 202105, 202104, 202103, 202102, 202101, 202012, 202011, 202010, 202002, 202001, 201912, 201911, 201910, 201909],
    "train": {
        "training": [202105, 202104, 202103, 202102, 202101, 202012, 202011, 202010, 202002, 202001, 201912, 201911, 201910, 201909, 201908, 201907],
        "validation": [202106],
        "testing": [202107],
        "undersampling": 0.2,
        "semilla": 700027,
    },
    "home": "~/buckets/b1/",
}

OUTPUT = {}

# Function to save output to a YAML file
def save_output():
    with open("output.yml", "w") as output_file:
        yaml.dump(OUTPUT, output_file)

# Set random seed
random.seed(PARAM["train"]["semilla"])

# Custom error handler
def error_handler():
    traceback(20)
    options(error=None)
    raise Exception("Exiting after script error")

options = {
    "error": error_handler
}

# Start time
OUTPUT["PARAM"] = PARAM
start_time = datetime.now()
OUTPUT["time"] = {"start": start_time.strftime("%Y%m%d %H%M%S")}

# Set working directory
os.chdir(os.path.expanduser(PARAM["home"]))

# Load the dataset
dataset_input = f"./exp/{PARAM['exp_input']}/dataset.csv.gz"
dataset = fread(dataset_input)

# Create experiment directory
exp_dir = f"./exp/{PARAM['experimento']}/"
os.makedirs(exp_dir, exist_ok=True)
os.chdir(exp_dir)

# Save output
save_output()
with open("parametros.yml", "w") as param_file:
    yaml.dump(PARAM, param_file)

# Order the dataset
dataset = dataset.sort_values(by=["foto_mes", "numero_de_cliente"])

# Save future data
future_data = dataset[dataset["foto_mes"].isin(PARAM["future"])]
future_data.to_csv("dataset_future.csv.gz", index=False, compression="gzip")

# Save training data for final models
train_final_data = dataset[dataset["foto_mes"].isin(PARAM["final_train"])]
train_final_data.to_csv("dataset_train_final.csv.gz", index=False, compression="gzip")

# Create a fold for training, validation, and testing
dataset["azar"] = np.random.uniform(size=len(dataset))
dataset["fold_train"] = 0
dataset.loc[(dataset["foto_mes"].isin(PARAM["train"]["training"])) & ((dataset["azar"] <= PARAM["train"]["undersampling"]) | (dataset["clase_ternaria"].isin(["BAJA+1", "BAJA+2"]))), "fold_train"] = 1

dataset["fold_validate"] = 0
dataset.loc[dataset["foto_mes"].isin(PARAM["train"]["validation"]), "fold_validate"] = 1

dataset["fold_test"] = 0
dataset.loc[dataset["foto_mes"].isin(PARAM["train"]["testing"]), "fold_test"] = 1

# Save the dataset for training
train_dataset = dataset[(dataset["fold_train"] + dataset["fold_validate"] + dataset["fold_test"] >= 1)]
train_dataset.to_csv("dataset_training.csv.gz", index=False, compression="gzip")

# Save dataset field information
tb_campos = pd.DataFrame({
    "pos": range(1, len(dataset.columns) + 1),
    "campo": dataset.columns,
    "tipo": dataset.dtypes.values,
    "nulos": train_dataset.isna().sum().values,
    "ceros": (train_dataset == 0).sum().values
})

tb_campos.to_csv("dataset_training.campos.txt", sep="\t", index=False)

# Save dataset information to OUTPUT
OUTPUT["dataset_train"] = {
    "ncol": len(train_dataset.columns),
    "nrow": len(train_dataset),
    "periodos": len(train_dataset["foto_mes"].unique())
}

OUTPUT["dataset_validate"] = {
    "ncol": len(dataset[dataset["fold_validate"] > 0].columns),
    "nrow": len(dataset[dataset["fold_validate"] > 0]),
    "periodos": len(dataset[dataset["fold_validate"] > 0]["foto_mes"].unique())
}

OUTPUT["dataset_test"] = {
    "ncol": len(dataset[dataset["fold_test"] > 0].columns),
    "nrow": len(dataset[dataset["fold_test"] > 0]),
    "periodos": len(dataset[dataset["fold_test"] > 0]["foto_mes"].unique())
}

OUTPUT["dataset_future"] = {
    "ncol": len(dataset[dataset["foto_mes"].isin(PARAM["future"])].columns),
    "nrow": len(dataset[dataset["foto_mes"].isin(PARAM["future"])]),
    "periodos": len(dataset[dataset["foto_mes"].isin(PARAM["future"])]["foto_mes"].unique())
}

OUTPUT["dataset_finaltrain"] = {
    "ncol": len(dataset[dataset["foto_mes"].isin(PARAM["final_train"])].columns),
    "nrow": len(dataset[dataset["foto_mes"].isin(PARAM["final_train"])]),
    "periodos": len(dataset[dataset["foto_mes"].isin(PARAM["final_train"])]["foto_mes"].unique())
}

# End time
end_time = datetime.now()
OUTPUT["time"]["end"] = end_time.strftime("%Y%m%d %H%M%S")

# Save output
save_output()

# Write final timestamp
with open("zRend.txt", "a") as timestamp_file:
    timestamp_file.write(end_time.strftime("%Y%m%d %H%M%S") + "\n")

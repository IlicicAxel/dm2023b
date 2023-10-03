import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
import time
from datetime import datetime
import os
import yaml
from data.table import fread
from rlist import setorder
from rollmean import frollmean

# Clear the memory
gc.collect()

# Define the parameters
PARAM = {
    "experimento": "HT6510",
    "exp_input": "TS6410",
    "lgb_crossvalidation_folds": 5,
    "lgb_semilla": 700027,
    "lgb_basicos": {
        "boosting": "gbdt",
        "objective": "binary",
        "metric": "custom",
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
        "max_depth": -1,
        "min_gain_to_split": 0.0,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_bin": 31,
        "num_iterations": 9999,
        "bagging_fraction": 1.0,
        "pos_bagging_fraction": 1.0,
        "neg_bagging_fraction": 1.0,
        "is_unbalance": False,
        "scale_pos_weight": 1.0,
        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,
        "extra_trees": True,
        "seed": 700027
    },
    "bo_lgb": {
        "learning_rate": (0.02, 0.3),
        "feature_fraction": (0.01, 1.0),
        "num_leaves": (8, 1024),
        "min_data_in_leaf": (100, 50000)
    },
    "bo_iteraciones": 50,
    "home": "~/buckets/b1/"
}

# Initialize OUTPUT as a dictionary
OUTPUT = {}

# Create a function to clear and record output
def clear_and_record_output():
    OUTPUT.clear()
    with open("output.yml", "w") as output_file:
        yaml.dump(OUTPUT, output_file)

# Record the start time
OUTPUT["PARAM"] = PARAM
OUTPUT["time"] = {"start": datetime.now().strftime("%Y%m%d %H%M%S")}

# Set the working directory
os.chdir(PARAM["home"])

# Load the dataset
dataset_input = f"./exp/{PARAM['exp_input']}/dataset_training.csv.gz"
dataset = fread(dataset_input)

# Create the experiment directory if it doesn't exist
exp_dir = f"./exp/{PARAM['experimento']}/"
os.makedirs(exp_dir, exist_ok=True)

# Set the working directory for the experiment
os.chdir(exp_dir)

# Create a function to record experiment logs
def exp_log(reg, arch=None, folder="./exp/", ext=".txt", verbose=True):
    archivo = arch if arch is not None else os.path.join(folder, str(reg) + ext)
    
    if not os.path.exists(archivo):
        linea = "fecha\t" + "\t".join(list(reg.keys())) + "\n"
        with open(archivo, "w") as f:
            f.write(linea)
    
    linea = datetime.now().strftime("%Y%m%d %H%M%S") + "\t" + "\t".join(map(str, reg.values())) + "\n"
    with open(archivo, "a") as f:
        f.write(linea)
    
    if verbose:
        print(linea, end="")

# Define the GLOBAL_arbol and GLOBAL_gan_max
GLOBAL_arbol = 0
GLOBAL_gan_max = -np.inf
vcant_optima = []

# Define the fganancia_lgbm_meseta function
def fganancia_lgbm_meseta(probs, datos):
    vlabels = datos["label"]
    vpesos = datos["weight"]

    global GLOBAL_arbol
    global GLOBAL_gan_max

    GLOBAL_arbol += 1
    tbl = pd.DataFrame({"prob": probs, "gan": np.where((vlabels == 1) & (vpesos > 1), 117000, -3000)})

    tbl = tbl.sort_values(by="prob", ascending=False)
    tbl["posicion"] = range(1, len(tbl) + 1)
    tbl["gan_acum"] = tbl["gan"].cumsum()

    tbl["gan_suavizada"] = frollmean(tbl["gan_acum"], n=2001, align="center", na_rm=True)

    gan = tbl["gan_suavizada"].max()
    pos = tbl["gan_suavizada"].idxmax()
    vcant_optima.append(pos)

    if GLOBAL_arbol % 10 == 0:
        if gan > GLOBAL_gan_max:
            GLOBAL_gan_max = gan

        print(f"\rValidate {GLOBAL_iteracion}   {GLOBAL_arbol}   {gan}   {GLOBAL_gan_max}   ", end="")

    return {"name": "ganancia", "value": gan, "higher_better": True}

# Define the EstimarGanancia_lightgbm function
def EstimarGanancia_lightgbm(x):
    global GLOBAL_arbol
    global GLOBAL_gan_max
    global vcant_optima

    gc.collect()
    GLOBAL_iteracion += 1
    OUTPUT["BO"]["iteracion_actual"] = GLOBAL_iteracion
    clear_and_record_output()

    param_completo = {**PARAM["lgb_basicos"], **x}
    param_completo["early_stopping_rounds"] = int(400 + 4 / param_completo["learning_rate"])

    GLOBAL_arbol = 0
    GLOBAL_gan_max = -np.inf
    vcant_optima = []

    np.random.seed(PARAM["lgb_semilla"])
    dtrain = lgb.Dataset(
        data=dataset[dataset["fold_train"] == 1][campos_buenos].values,
        label=dataset[dataset["fold_train"] == 1]["clase01"].values,
        weight=np.where(dataset[dataset["fold_train"] == 1]["clase_ternaria"] == "BAJA+2", 1.0000001, 1.0),
        free_raw_data=False
    )

    print("\n")

    modelo_train = lgb.train(
        train_set=dtrain,
        valid_sets=[(dtrain, "train")],
        feval=fganancia_lgbm_meseta,
        params=param_completo,
        verbose_eval=-100
    )

    cant_corte = vcant_optima[modelo_train.best_iteration - 1]

    dtest = dataset_test[campos_buenos].values
    prediccion = modelo_train.predict(dtest, num_iteration=modelo_train.best_iteration)

    tbl = pd.DataFrame({"gan": np.where(dataset_test["clase_ternaria"] == "BAJA+2", 117000, -3000)})
    tbl["prob"] = prediccion
    tbl = tbl.sort_values(by="prob", ascending=False)
    tbl["gan_acum"] = tbl["gan"].cumsum()
    tbl["gan_suavizada"] = frollmean(tbl["gan_acum"], n=2001, align="center", na_rm=True)

    ganancia_test = tbl["gan_suavizada"].max()
    cantidad_test_normalizada = tbl["gan_suavizada"].idxmax()

    ganancia_test_normalizada = ganancia_test

    if ganancia_test_normalizada > GLOBAL_ganancia:
        GLOBAL_ganancia = ganancia_test_normalizada

        param_impo = param_completo.copy()
        param_impo["early_stopping_rounds"] = 0
        param_impo["num_iterations"] = modelo_train.best_iteration

        modelo = lgb.train(
            train_set=dtrain,
            params=param_impo,
            verbose_eval=-100
        )

        tb_importancia = pd.DataFrame(modelo.feature_importance(), columns=["Importance"], index=campos_buenos)
        tb_importancia.to_csv(f"impo_{GLOBAL_iteracion:03d}.txt", sep="\t")

        OUTPUT["BO"]["mejor"]["iteracion"] = GLOBAL_iteracion
        OUTPUT["BO"]["mejor"]["ganancia"] = GLOBAL_ganancia
        OUTPUT["BO"]["mejor"]["arboles"] = modelo_train.best_iteration
        clear_and_record_output()

    ds = {"cols": dtrain.num_feature(), "rows": dtrain.num_data()}
    xx = {**ds, **param_completo}
    xx["early_stopping_rounds"] = None
    xx["num_iterations"] = modelo_train.best_iteration
    xx["estimulos"] = cantidad_test_normalizada
    xx["ganancia"] = ganancia_test_normalizada
    xx["iteracion_bayesiana"] = GLOBAL_iteracion

    exp_log(xx, arch="BO_log.txt")
    np.random.seed(PARAM["lgb_semilla"])

    return ganancia_test_normalizada

# Define the GLOBAL_arbol and GLOBAL_gan_max for cross-validation
GLOBAL_arbol = 0
GLOBAL_gan_max = -np.inf

# Define the fganancia_lgbm_mesetaCV function for cross-validation
def fganancia_lgbm_mesetaCV(probs, datos):
    vlabels = datos["label"]
    vpesos = datos["weight"]

    global GLOBAL_arbol
    global GLOBAL_gan_max

    GLOBAL_arbol += 1

    tbl = pd.DataFrame({"prob": probs, "gan": np.where((vlabels == 1) & (vpesos > 1), 117000, -3000)})

    tbl = tbl.sort_values(by="prob", ascending=False)
    tbl["posicion"] = range(1, len(tbl) + 1)
    tbl["gan_acum"] = tbl["gan"].cumsum()
    tbl["gan_suavizada"] = frollmean(tbl["gan_acum"], n=501, align="center", na_rm=True)

    gan = tbl["gan_suavizada"].max()
    pos = tbl["gan_suavizada"].idxmax()
    vcant_optima.append(pos)

    if GLOBAL_arbol % (10 * PARAM["lgb_crossvalidation_folds"]) == 0:
        if gan > GLOBAL_gan_max:
            GLOBAL_gan_max = gan

        print(f"\rCross Validate {GLOBAL_iteracion}   {int(GLOBAL_arbol / PARAM['lgb_crossvalidation_folds'])}   {gan * PARAM['lgb_crossvalidation_folds']}   {GLOBAL_gan_max * PARAM['lgb_crossvalidation_folds']}   ", end="")

    return {"name": "ganancia", "value": gan, "higher_better": True}

# Define the EstimarGanancia_lightgbmCV function for cross-validation
def EstimarGanancia_lightgbmCV(x):
    global GLOBAL_arbol
    global GLOBAL_gan_max

    gc.collect()
    GLOBAL_iteracion += 1
    OUTPUT["BO"]["iteracion_actual"] = GLOBAL_iteracion
    clear_and_record_output()

    param_completo = {**PARAM["lgb_basicos"], **x}
    param_completo["early_stopping_rounds"] = int(400 + 4 / param_completo["learning_rate"])

    vcant_optima = []
    GLOBAL_arbol = 0
    GLOBAL_gan_max = -np.inf

    np.random.seed(PARAM["lgb_semilla"])
    dtrain = lgb.Dataset(
        data=dataset[dataset["fold_train"] == 1][campos_buenos].values,
        label=dataset[dataset["fold_train"] == 1]["clase01"].values,
        weight=np.where(dataset[dataset["fold_train"] == 1]["clase_ternaria"] == "BAJA+2", 1.0000001, 1.0),
        free_raw_data=False
    )

    print("\n")

    modelocv = lgb.cv(
        train_set=dtrain,
        params=param_completo,
        feval=fganancia_lgbm_mesetaCV,
        num_boost_round=9999,
        nfold=PARAM["lgb_crossvalidation_folds"],
        verbose_eval=-100,
        early_stopping_rounds=None
    )

    desde = (modelocv["best_iteration"] - 1) * PARAM["lgb_crossvalidation_folds"] + 1
    hasta = desde + PARAM["lgb_crossvalidation_folds"] - 1

    cant_corte = int(np.mean(vcant_optima[desde:hasta]) * PARAM["lgb_crossvalidation_folds"])

    ganancia = modelocv["valid_0"]["ganancia"][modelocv["best_iteration"] - 1]
    ganancia_normalizada = ganancia * PARAM["lgb_crossvalidation_folds"]

    if ktest:
        param_completo["early_stopping_rounds"] = None
        param_completo["num_iterations"] = modelocv["best_iteration"]

        modelo = lgb.train(
            train_set=dtrain,
            params=param_completo,
            verbose_eval=-100
        )

        dtest = dataset_test[campos_buenos_test].values
        prediccion = modelo.predict(dtest, num_iteration=modelo.best_iteration)

        tbl = pd.DataFrame({"gan": np.where(dataset_test["clase_ternaria"] == "BAJA+2", 117000, -3000)})
        tbl["prob"] = prediccion
        tbl = tbl.sort_values(by="prob", ascending=False)
        tbl["gan_acum"] = tbl["gan"].cumsum()
        tbl["gan_suavizada"] = frollmean(tbl["gan_acum"], n=2001, align="center", na_rm=True)

        ganancia_test_normalizada = tbl["gan_suavizada"].max()
        cant_corte = tbl["gan_suavizada"].idxmax()

    if ganancia_normalizada > GLOBAL_ganancia:
        GLOBAL_ganancia = ganancia_normalizada

        param_impo = param_completo.copy()
        param_impo["early_stopping_rounds"] = 0
        param_impo["num_iterations"] = modelocv["best_iteration"]

        modelo = lgb.train(
            train_set=dtrain,
            params=param_impo,
            verbose_eval=-100
        )

        tb_importancia = pd.DataFrame(modelo.feature_importance(), columns=["Importance"], index=campos_buenos)
        tb_importancia.to_csv(f"impo_{GLOBAL_iteracion}.txt", sep="\t")

        OUTPUT["BO"]["mejor"]["iteracion"] = GLOBAL_iteracion
        OUTPUT["BO"]["mejor"]["ganancia"] = GLOBAL_ganancia
        OUTPUT["BO"]["mejor"]["arboles"] = modelo.best_iteration
        clear_and_record_output()

    ds = {"cols": dtrain.num_feature(), "rows": dtrain.num_data()}
    xx = {**ds, **param_completo}
    xx["early_stopping_rounds"] = None
    xx["num_iterations"] = modelocv["best_iteration"]
    xx["estimulos"] = cant_corte
    xx["ganancia"] = ganancia_normalizada
    xx["iteracion_bayesiana"] = GLOBAL_iteracion

    exp_log(xx, arch="BO_log.txt")
    np.random.seed(PARAM["lgb_semilla"])

    return ganancia_normalizada

# Define the main program
if __name__ == "__main__":
    OUTPUT["PARAM"] = PARAM
    OUTPUT["time"]["start"] = datetime.now().strftime("%Y%m%d %H%M%S")

    os.chdir(PARAM["home"])

    dataset_input = f"./exp/{PARAM['exp_input']}/dataset_training.csv.gz"
    dataset = pd.read_csv(dataset_input)

    dataset.drop(columns=["azar"], inplace=True)

    if "fold_train" not in dataset.columns:
        raise ValueError("Error, el dataset no tiene el campo fold_train")

    if "fold_validate" not in dataset.columns:
        raise ValueError("Error, el dataset no tiene el campo fold_validate")

    if "fold_test" not in dataset.columns:
        raise ValueError("Error, el dataset no tiene el campo fold_test")

    if dataset[dataset["fold_train"] == 1].shape[0] == 0:
        raise ValueError("Error, en el dataset no hay registros con fold_train==1")

    os.makedirs(f"./exp/{PARAM['experimento']}/", exist_ok=True)
    os.chdir(f"./exp/{PARAM['experimento']}/")

    clear_and_record_output()

    with open("parametros.yml", "w") as f:
        yaml.dump(PARAM, f)

    with open("TrainingStrategy.txt", "w") as f:
        f.write(PARAM["exp_input"])

    dataset["clase01"] = np.where(dataset["clase_ternaria"] == "CONTINUA", 0, 1)

    campos_buenos = dataset.columns.difference(["clase01", "clase_ternaria", "fold_train", "fold_validate", "fold_test"])

    dtrain = lgb.Dataset(
        data=dataset[dataset["fold_train"] == 1][campos_buenos].values,
        label=dataset[dataset["fold_train"] == 1]["clase01"].values,
        weight=np.where(dataset[dataset["fold_train"] == 1]["clase_ternaria"] == "BAJA+2", 1.0000001, 1.0),
        free_raw_data=False
    )

    OUTPUT["train"]["ncol"] = dtrain.num_feature()
    OUTPUT["train"]["nrow"] = dtrain.num_data()
    OUTPUT["train"]["periodos"] = dataset[dataset["fold_train"] == 1]["foto_mes"].nunique()

    kvalidate = False
    ktest = False
    kcrossvalidation = True

    if dataset[(dataset["fold_train"] == 0) & (dataset["fold_test"] == 0) & (dataset["fold_validate"] == 1)].shape[0] > 0:
        kcrossvalidation = False
        kvalidate = True
        dvalidate = lgb.Dataset(
            data=dataset[dataset["fold_validate"] == 1][campos_buenos].values,
            label=dataset[dataset["fold_validate"] == 1]["clase01"].values,
            weight=np.where(dataset[dataset["fold_validate"] == 1]["clase_ternaria"] == "BAJA+2", 1.0000001, 1.0),
            free_raw_data=False
        )

        OUTPUT["validate"]["ncol"] = dvalidate.num_feature()
        OUTPUT["validate"]["nrow"] = dvalidate.num_data()
        OUTPUT["validate"]["periodos"] = dataset[dataset["fold_validate"] == 1]["foto_mes"].nunique()

    if dataset[(dataset["fold_train"] == 0) & (dataset["fold_validate"] == 0) & (dataset["fold_test"] == 1)].shape[0] > 0:
        ktest = True
        campos_buenos_test = dataset.columns.difference(["clase01", "clase_ternaria", "fold_train", "fold_validate", "fold_test"])

        dataset_test = dataset[dataset["fold_test"] == 1]

        campos_buenos_test = dataset_test.columns.difference(["clase01", "clase_ternaria", "fold_train", "fold_validate", "fold_test"])

    if kcrossvalidation:
        resultados_crossvalidation = []
        train0 = dataset[dataset["fold_train"] == 0].copy()
        train0 = train0.sample(frac=1, random_state=700027)

        for i in range(PARAM["lgb_crossvalidation_folds"]):
            fold = i + 1
            data_train = train0[train0["fold_validate"] != fold]

            np.random.seed(PARAM["lgb_semilla"])
            dtrain = lgb.Dataset(
                data=data_train[campos_buenos].values,
                label=data_train["clase01"].values,
                weight=np.where(data_train["clase_ternaria"] == "BAJA+2", 1.0000001, 1.0),
                free_raw_data=False
            )

            data_validate = train0[train0["fold_validate"] == fold]
            dvalidate = lgb.Dataset(
                data=data_validate[campos_buenos].values,
                label=data_validate["clase01"].values,
                weight=np.where(data_validate["clase_ternaria"] == "BAJA+2", 1.0000001, 1.0),
                free_raw_data=False
            )

            OUTPUT["train"]["ncol"] = dtrain.num_feature()
            OUTPUT["train"]["nrow"] = dtrain.num_data()
            OUTPUT["train"]["periodos"] = data_train["foto_mes"].nunique()

            OUTPUT["validate"]["ncol"] = dvalidate.num_feature()
            OUTPUT["validate"]["nrow"] = dvalidate.num_data()
            OUTPUT["validate"]["periodos"] = data_validate["foto_mes"].nunique()

            p = EstimarGanancia_lightgbmCV(x)
            resultados_crossvalidation.append(p)

            print(f"\n**** Cross-Validation {fold}/{PARAM['lgb_crossvalidation_folds']}   Ganancia: {p} ****\n")

        ganancia_promedio_crossvalidation = np.mean(resultados_crossvalidation)
        ganancia_desvio_crossvalidation = np.std(resultados_crossvalidation)
        print(f"**** Cross-Validation Final   Ganancia Promedio: {ganancia_promedio_crossvalidation}   Desvío: {ganancia_desvio_crossvalidation} ****\n")

    OUTPUT["time"]["end"] = datetime.now().strftime("%Y%m%d %H%M%S")

    clear_and_record_output()

    if ganancia_promedio_crossvalidation > 0:
        dtrain = lgb.Dataset(
            data=dataset[dataset["fold_train"] == 1][campos_buenos].values,
            label=dataset[dataset["fold_train"] == 1]["clase01"].values,
            weight=np.where(dataset[dataset["fold_train"] == 1]["clase_ternaria"] == "BAJA+2", 1.0000001, 1.0),
            free_raw_data=False
        )

        OUTPUT["train"]["ncol"] = dtrain.num_feature()
        OUTPUT["train"]["nrow"] = dtrain.num_data()
        OUTPUT["train"]["periodos"] = dataset[dataset["fold_train"] == 1]["foto_mes"].nunique()

        OUTPUT["validate"]["ncol"] = dvalidate.num_feature()
        OUTPUT["validate"]["nrow"] = dvalidate.num_data()
        OUTPUT["validate"]["periodos"] = dataset[dataset["fold_validate"] == 1]["foto_mes"].nunique()

        OUTPUT["time"]["start"] = datetime.now().strftime("%Y%m%d %H%M%S")

        modelo = lgb.train(
            train_set=dtrain,
            params=param_impo,
            verbose_eval=-100
        )

        tb_importancia = pd.DataFrame(modelo.feature_importance(), columns=["Importance"], index=campos_buenos)
        tb_importancia.to_csv("impo_000.txt", sep="\t")

        dtest = dataset_test[campos_buenos_test].values
        prediccion = modelo.predict(dtest, num_iteration=modelo.best_iteration)

        tbl = pd.DataFrame({"gan": np.where(dataset_test["clase_ternaria"] == "BAJA+2", 117000, -3000)})
        tbl["prob"] = prediccion
        tbl = tbl.sort_values(by="prob", ascending=False)
        tbl["gan_acum"] = tbl["gan"].cumsum()
        tbl["gan_suavizada"] = frollmean(tbl["gan_acum"], n=2001, align="center", na_rm=True)

        ganancia_test = tbl["gan_suavizada"].max()
        cantidad_test_normalizada = tbl["gan_suavizada"].idxmax()

        OUTPUT["time"]["end"] = datetime.now().strftime("%Y%m%d %H%M%S")

        clear_and_record_output()

        print("\n**** Test Final   Ganancia: " + str(ganancia_test) + " ****\n")

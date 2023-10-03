import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from random import randint
import gc

# Parameters
PARAM = {
    "experimento": "FE6310",
    "exp_input": "DR6210",
    "lag1": True,
    "lag2": True,
    "lag3": False,
    "Tendencias1": {
        "run": True,
        "ventana": 3,
        "tendencia": False,
        "minimo": False,
        "maximo": False,
        "promedio": True,
        "ratioavg": False,
        "ratiomax": False,
    },
    "Tendencias2": {
        "run": True,
        "ventana": 6,
        "tendencia": True,
        "minimo": False,
        "maximo": False,
        "promedio": False,
        "ratioavg": False,
        "ratiomax": False,
    },
    "RandomForest": {
        "run": True,
        "num.trees": 20,
        "max.depth": 4,
        "min.node.size": 1000,
        "mtry": 40,
        "semilla": 700027,
    },
    "CanaritosAsesinos": {
        "ratio": 0.1,
        "desvios": 4.0,
        "semilla": 700057,
    },
    "home": "~/buckets/b1/",
}

OUTPUT = {}

# Function to save output to a YAML file
def save_output():
    with open("output.yml", "w") as output_file:
        yaml.dump(OUTPUT, output_file)

# Custom loss function for LGBM
def fganancia_lgbm_meseta(probs, datos):
    vlabels = get_field(datos, "label")
    vpesos = get_field(datos, "weight")

    tbl = pd.DataFrame({
        "prob": probs,
        "gan": np.where((vlabels == 1) & (vpesos > 1), 117000, -3000)
    })

    tbl = tbl.sort_values(by="prob", ascending=False)
    tbl["posicion"] = np.arange(1, len(tbl) + 1)
    tbl["gan_acum"] = tbl["gan"].cumsum()

    gan = tbl.loc[:499, "gan_acum"].mean()

    pos_meseta = tbl.loc[:499, "posicion"].median()
    VPOS_CORTE.append(pos_meseta)

    return {
        "name": "ganancia",
        "value": gan,
        "higher_better": True
    }

# Custom RcppFunction for computing trends
def fhistC(pcolumna, pdesde):
    x = np.zeros(100)
    y = np.zeros(100)

    n = len(pcolumna)
    out = np.zeros(5 * n)

    for i in range(n):
        if pdesde[i] - 1 < i:
            out[i + 4 * n] = pcolumna[i - 1]
        else:
            out[i + 4 * n] = np.nan

        libre = 0
        xvalor = 1

        for j in range(pdesde[i] - 1, i + 1):
            a = pcolumna[j]

            if not np.isnan(a):
                y[libre] = a
                x[libre] = xvalor
                libre += 1

            xvalor += 1

        if libre > 1:
            xsum = x[0]
            ysum = y[0]
            xysum = xsum * ysum
            xxsum = xsum * xsum
            vmin = y[0]
            vmax = y[0]

            for h in range(1, libre):
                xsum += x[h]
                ysum += y[h]
                xysum += x[h] * y[h]
                xxsum += x[h] * x[h]

                if y[h] < vmin:
                    vmin = y[h]

                if y[h] > vmax:
                    vmax = y[h]

            out[i] = (libre * xysum - xsum * ysum) / (libre * xxsum - xsum * xsum)
            out[i + n] = vmin
            out[i + 2 * n] = vmax
            out[i + 3 * n] = ysum / libre
        else:
            out[i] = np.nan
            out[i + n] = np.nan
            out[i + 2 * n] = np.nan
            out[i + 3 * n] = np.nan

    return out.tolist()

# Function to calculate trends for columns
def TendenciaYmuchomas(dataset, cols, ventana=6, tendencia=True, minimo=True, maximo=True, promedio=True,
                        ratioavg=False, ratiomax=False):
    gc.collect()
    ventana_regresion = ventana
    last = len(dataset)
    vector_ids = dataset["numero_de_cliente"].values
    vector_desde = np.full(last, -ventana_regresion + 2)

    for i in range(1, last):
        if vector_ids[i - 1] != vector_ids[i]:
            vector_desde[i] = i

    for i in range(1, last):
        if vector_desde[i] < vector_desde[i - 1]:
            vector_desde[i] = vector_desde[i - 1]

    for campo in cols:
        nueva_col = fhistC(dataset[campo].values, vector_desde)

        if tendencia:
            dataset[f"{campo}_tend{ventana}"] = nueva_col[0 * last:1 * last]

        if minimo:
            dataset[f"{campo}_min{ventana}"] = nueva_col[1 * last:2 * last]

        if maximo:
            dataset[f"{campo}_max{ventana}"] = nueva_col[2 * last:3 * last]

        if promedio:
            dataset[f"{campo}_avg{ventana}"] = nueva_col[3 * last:4 * last]

        if ratioavg:
            dataset[f"{campo}_ratioavg{ventana}"] = dataset[campo] / nueva_col[3 * last:4 * last]

        if ratiomax:
            dataset[f"{campo}_ratiomax{ventana}"] = dataset[campo] / nueva_col[2 * last:3 * last]

# Function to add variables based on Random Forest leaf nodes
def AgregaVarRandomForest(num_trees, max_depth, min_node_size, mtry, semilla):
    gc.collect()
    dataset["clase01"] = np.where(dataset["clase_ternaria"] == "CONTINUA", 0, 1)
    campos_buenos = dataset.columns.difference(["clase_ternaria"])
    dataset_rf = dataset[campos_buenos].copy()
    np.random.seed(semilla)
    azar = np.random.uniform(size=len(dataset_rf))
    dataset_rf["azar"] = azar
    train = dataset_rf.loc[dataset_rf["azar"] < 0.5].copy()
    valid = dataset_rf.loc[(dataset_rf["azar"] >= 0.5) & (dataset_rf["azar"] < 0.75)].copy()
    test = dataset_rf.loc[dataset_rf["azar"] >= 0.75].copy()

    variables_prediccion = ["clase01"]

    variables_random_forest = [x for x in campos_buenos if x not in variables_prediccion]

    lgb_train = lgb.Dataset(train[variables_random_forest], train[variables_prediccion], free_raw_data=False)
    lgb_valid = lgb.Dataset(valid[variables_random_forest], valid[variables_prediccion], reference=lgb_train, free_raw_data=False)
    lgb_test = lgb.Dataset(test[variables_random_forest], test[variables_prediccion], reference=lgb_train, free_raw_data=False)
    
    params = {
        "num_trees": num_trees,
        "objective": "binary",
        "max_depth": max_depth,
        "min_data_in_leaf": min_node_size,
        "num_leaves": 2 ** max_depth,
        "learning_rate": 0.01,
        "feature_fraction": mtry / len(variables_random_forest),
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=200,
        verbose_eval=100,
        feval=fganancia_lgbm_meseta,
    )

    train["leaves"] = model.predict(train[variables_random_forest], pred_leaf=True).argmax(axis=0)
    valid["leaves"] = model.predict(valid[variables_random_forest], pred_leaf=True).argmax(axis=0)
    test["leaves"] = model.predict(test[variables_random_forest], pred_leaf=True).argmax(axis=0)
    leaves = set(train["leaves"].unique()).union(set(valid["leaves"].unique())).union(set(test["leaves"].unique()))

    train = pd.get_dummies(train, columns=["leaves"])
    valid = pd.get_dummies(valid, columns=["leaves"])
    test = pd.get_dummies(test, columns=["leaves"])

    return train, valid, test

# Main function
if __name__ == "__main__":
    os.chdir(os.path.expanduser(PARAM["home"]))
    
    # Load dataset
    dataset = pd.read_csv(f"{PARAM['exp_input']}.csv")

    # Generate lag features
    if PARAM["lag1"]:
        dataset["clase_lag1"] = dataset.groupby("numero_de_cliente")["clase01"].shift(1)

    if PARAM["lag2"]:
        dataset["clase_lag2"] = dataset.groupby("numero_de_cliente")["clase01"].shift(2)

    if PARAM["lag3"]:
        dataset["clase_lag3"] = dataset.groupby("numero_de_cliente")["clase01"].shift(3)

    # Generate trend features
    if PARAM["Tendencias1"]["run"]:
        TendenciaYmuchomas(
            dataset,
            ["saldo_adeudado_pesos", "saldo_adeudado_dolar", "adelanto_efectivo_dolar"],
            ventana=PARAM["Tendencias1"]["ventana"],
            tendencia=PARAM["Tendencias1"]["tendencia"],
            minimo=PARAM["Tendencias1"]["minimo"],
            maximo=PARAM["Tendencias1"]["maximo"],
            promedio=PARAM["Tendencias1"]["promedio"],
            ratioavg=PARAM["Tendencias1"]["ratioavg"],
            ratiomax=PARAM["Tendencias1"]["ratiomax"]
        )

    if PARAM["Tendencias2"]["run"]:
        TendenciaYmuchomas(
            dataset,
            ["saldo_adeudado_pesos", "saldo_adeudado_dolar", "adelanto_efectivo_dolar"],
            ventana=PARAM["Tendencias2"]["ventana"],
            tendencia=PARAM["Tendencias2"]["tendencia"],
            minimo=PARAM["Tendencias2"]["minimo"],
            maximo=PARAM["Tendencias2"]["maximo"],
            promedio=PARAM["Tendencias2"]["promedio"],
            ratioavg=PARAM["Tendencias2"]["ratioavg"],
            ratiomax=PARAM["Tendencias2"]["ratiomax"]
        )

    # Generate additional features based on Random Forest
    if PARAM["RandomForest"]["run"]:
        train, valid, test = AgregaVarRandomForest(
            num_trees=PARAM["RandomForest"]["num.trees"],
            max_depth=PARAM["RandomForest"]["max.depth"],
            min_node_size=PARAM["RandomForest"]["min.node.size"],
            mtry=PARAM["RandomForest"]["mtry"],
            semilla=PARAM["RandomForest"]["semilla"]
        )

        dataset = pd.concat([dataset, train, valid, test], axis=0, ignore_index=True)

    # Save the final dataset to a CSV file
    dataset.to_csv(f"{PARAM['experimento']}_final.csv", index=False)

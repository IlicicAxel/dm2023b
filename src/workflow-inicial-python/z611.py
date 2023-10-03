import yaml
import pandas as pd
import os

# Parametros del script
PARAM = {}
PARAM['experimento'] = 'CA6110'
PARAM['dataset'] = './datasets/competencia_2023.csv.gz'

# valores posibles
#  "MachineLearning"  "EstadisticaClasica" "Ninguno"
PARAM['metodo'] = 'EstadisticaClasica'
#Set PARAM['home'] in bucket/b1 in the root of the project

PARAM['home'] = '/home/ailicicisely/buckets/b1/'

# FIN Parametros del script

OUTPUT = {}

#------------------------------------------------------------------------------

def GrabarOutput():
    with open('output.yml', 'w') as f:
        yaml.dump(OUTPUT, f) # grabo output

#------------------------------------------------------------------------------

def CorregirCampoMes(pcampo, pmeses, dataset):
    tbl = dataset.groupby('numero_de_cliente').apply(lambda x: pd.DataFrame({
        'v1': x[pcampo].shift(1),
        'v2': x[pcampo].shift(-1)
    }))
    tbl.reset_index(level=1, drop=True, inplace=True)
    tbl['promedio'] = tbl.mean(axis=1, skipna=True)

    dataset.loc[~dataset['foto_mes'].isin(pmeses), pcampo] = \
        dataset[~dataset['foto_mes'].isin(pmeses)][pcampo]
    dataset.loc[dataset['foto_mes'].isin(pmeses), pcampo] = \
        tbl.loc[dataset['foto_mes'].isin(pmeses), 'promedio'].values
    

def Corregir_EstadisticaClasica(dataset):
    CorregirCampoMes(dataset, "thomebanking", [201801, 202006])
    CorregirCampoMes(dataset, "chomebanking_transacciones", [201801, 201910, 202006])
    CorregirCampoMes(dataset, "tcallcenter", [201801, 201806, 202006])
    CorregirCampoMes(dataset, "ccallcenter_transacciones", [201801, 201806, 202006])
    CorregirCampoMes(dataset, "cprestamos_personales", [201801, 202006])
    CorregirCampoMes(dataset, "mprestamos_personales", [201801, 202006])
    CorregirCampoMes(dataset, "mprestamos_hipotecarios", [201801, 202006])
    CorregirCampoMes(dataset, "ccajas_transacciones", [201801, 202006])
    CorregirCampoMes(dataset, "ccajas_consultas", [201801, 202006])
    CorregirCampoMes(dataset, "ccajas_depositos", [201801, 202006])
    CorregirCampoMes(dataset, "ccajas_extracciones", [201801, 202006])
    CorregirCampoMes(dataset, "ccajas_otras", [201801, 202006])

    CorregirCampoMes(dataset, "ctarjeta_visa_debitos_automaticos", [201904])
    CorregirCampoMes(dataset, "mttarjeta_visa_debitos_automaticos", [201904, 201905])
    CorregirCampoMes(dataset, "Visa_mfinanciacion_limite", [201904])

    CorregirCampoMes(dataset, "mrentabilidad", [201905, 201910, 202006])
    CorregirCampoMes(dataset, "mrentabilidad_annual", [201905, 201910, 202006])
    CorregirCampoMes(dataset, "mcomisiones", [201905, 201910, 202006])
    CorregirCampoMes(dataset, "mpasivos_margen", [201905, 201910, 202006])
    CorregirCampoMes(dataset, "mactivos_margen", [201905, 201910, 202006])
    CorregirCampoMes(dataset, "ccomisiones_otras", [201905, 201910, 202006])
    CorregirCampoMes(dataset, "mcomisiones_otras", [201905, 201910, 202006])

    CorregirCampoMes(dataset, "ctarjeta_visa_descuentos", [201910])
    CorregirCampoMes(dataset, "ctarjeta_master_descuentos", [201910])
    CorregirCampoMes(dataset, "mtarjeta_visa_descuentos", [201910])
    CorregirCampoMes(dataset, "mtarjeta_master_descuentos", [201910])
    CorregirCampoMes(dataset, "ccajeros_propios_descuentos", [201910])
    CorregirCampoMes(dataset, "mcajeros_propios_descuentos", [201910])

    CorregirCampoMes(dataset, "cliente_vip", [201911])

    CorregirCampoMes(dataset, "active_quarter", [202006])
    CorregirCampoMes(dataset, "mcuentas_saldo", [202006])
    CorregirCampoMes(dataset, "ctarjeta_debito_transacciones", [202006])
    CorregirCampoMes(dataset, "mautoservicio", [202006])
    CorregirCampoMes(dataset, "ctarjeta_visa_transacciones", [202006])
    CorregirCampoMes(dataset, "ctarjeta_visa_transacciones", [202006])
    CorregirCampoMes(dataset, "cextraccion_autoservicio", [202006])
    CorregirCampoMes(dataset, "mextraccion_autoservicio", [202006])
    CorregirCampoMes(dataset, "ccheques_depositados", [202006])
    CorregirCampoMes(dataset, "mcheques_depositados", [202006])
    CorregirCampoMes(dataset, "mcheques_emitidos", [202006])
    CorregirCampoMes(dataset, "mcheques_emitidos", [202006])
    CorregirCampoMes(dataset, "ccheques_depositados_rechazados", [202006])
    CorregirCampoMes(dataset, "mcheques_depositados_rechazados", [202006])
    CorregirCampoMes(dataset, "ccheques_emitidos_rechazados", [202006])
    CorregirCampoMes(dataset, "mcheques_emitidos_rechazados", [202006])
    CorregirCampoMes(dataset, "catm_trx", [202006])
    CorregirCampoMes(dataset, "matm", [202006])
    CorregirCampoMes(dataset, "catm_trx_other", [202006])
    CorregirCampoMes(dataset, "matm_other", [202006])

def CorregirCampoMes(dataset, variable, meses):
    for mes in meses:
        mes_anterior = mes - 1
        mes_posterior = mes + 1
        dataset.loc[dataset.foto_mes == mes, variable] = (dataset.loc[dataset.foto_mes == mes_anterior, variable] + dataset.loc[dataset.foto_mes == mes_posterior, variable]) / 2

def Corregir_MachineLearning(dataset):
    dataset.loc[dataset.foto_mes == 201901, "ctransferencias_recibidas"] = None
    dataset.loc[dataset.foto_mes == 201901, "mtransferencias_recibidas"] = None

    dataset.loc[dataset.foto_mes == 201902, "ctransferencias_recibidas"] = None
    dataset.loc[dataset.foto_mes == 201902, "mtransferencias_recibidas"] = None

    dataset.loc[dataset.foto_mes == 201903, "ctransferencias_recibidas"] = None
    dataset.loc[dataset.foto_mes == 201903, "mtransferencias_recibidas"] = None

    dataset.loc[dataset.foto_mes == 201904, "ctransferencias_recibidas"] = None
    dataset.loc[dataset.foto_mes == 201904, "mtransferencias_recibidas"] = None
    dataset.loc[dataset.foto_mes == 201904, "ctarjeta_visa_debitos_automaticos"] = None
    dataset.loc[dataset.foto_mes == 201904, "mttarjeta_visa_debitos_automaticos"] = None
    dataset.loc[dataset.foto_mes == 201904, "Visa_mfinanciacion_limite"] = None

    dataset.loc[dataset.foto_mes == 201905, "ctransferencias_recibidas"] = None
    dataset.loc[dataset.foto_mes == 201905, "mtransferencias_recibidas"] = None
    dataset.loc[dataset.foto_mes == 201905, "mrentabilidad"] = None
    dataset.loc[dataset.foto_mes == 201905, "mrentabilidad_annual"] = None
    dataset.loc[dataset.foto_mes == 201905, "mcomisiones"] = None
    dataset.loc[dataset.foto_mes == 201905, "mpasivos_margen"] = None
    dataset.loc[dataset.foto_mes == 201905, "mactivos_margen"] = None
    dataset.loc[dataset.foto_mes == 201905, "ctarjeta_visa_debitos_automaticos"] = None
    dataset.loc[dataset.foto_mes == 201905, "ccomisiones_otras"] = None
    dataset.loc[dataset.foto_mes == 201905, "mcomisiones_otras"] = None

    dataset.loc[dataset.foto_mes == 201910, "mpasivos_margen"] = None
    dataset.loc[dataset.foto_mes == 201910, "mactivos_margen"] = None
    dataset.loc[dataset.foto_mes == 201910, "ccomisiones_otras"] = None
    dataset.loc[dataset.foto_mes == 201910, "mcomisiones_otras"] = None
    dataset.loc[dataset.foto_mes == 201910, "mcomisiones"] = None
    dataset.loc[dataset.foto_mes == 201910, "mrentabilidad"] = None
    dataset.loc[dataset.foto_mes == 201910, "mrentabilidad_annual"] = None
    dataset.loc[dataset.foto_mes == 201910, "chomebanking_transacciones"] = None
    dataset.loc[dataset.foto_mes == 201910, "ctarjeta_visa_descuentos"] = None
    dataset.loc[dataset.foto_mes == 201910, "ctarjeta_master_descuentos"] = None
    dataset.loc[dataset.foto_mes == 201910, "mtarjeta_visa_descuentos"] = None
    dataset.loc[dataset.foto_mes == 201910, "mtarjeta_master_descuentos"] = None
    dataset.loc[dataset.foto_mes == 201910, "ccajeros_propios_descuentos"] = None
    dataset.loc[dataset.foto_mes == 201910, "mcajeros_propios_descuentos"] = None

    dataset.loc[dataset.foto_mes == 202001, "cliente_vip"] = None

    dataset.loc[dataset.foto_mes == 202006, "active_quarter"] = None
    dataset.loc[dataset.foto_mes == 202006, "mrentabilidad"] = None
    dataset.loc[dataset.foto_mes == 202006, "mrentabilidad_annual"] = None
    dataset.loc[dataset.foto_mes == 202006, "mcomisiones"] = None
    dataset.loc[dataset.foto_mes == 202006, "mactivos_margen"] = None
    dataset.loc[dataset.foto_mes == 202006, "mpasivos_margen"] = None
    dataset.loc[dataset.foto_mes == 202006, "mcuentas_saldo"] = None
    dataset.loc[dataset.foto_mes == 202006, "ctarjeta_debito_transacciones"] = None
    dataset.loc[dataset.foto_mes == 202006, "mautoservicio"] = None
    dataset.loc[dataset.foto_mes == 202006, "ctarjeta_visa_transacciones"] = None
    dataset.loc[dataset.foto_mes == 202006, "mtarjeta_visa_consumo"] = None
    dataset.loc[dataset.foto_mes == 202006, "ctarjeta_master_transacciones"] = None
    dataset.loc[dataset.foto_mes == 202006, "mtarjeta_master_consumo"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccomisiones_otras"] = None
    dataset.loc[dataset.foto_mes == 202006, "mcomisiones_otras"] = None
    dataset.loc[dataset.foto_mes == 202006, "cextraccion_autoservicio"] = None
    dataset.loc[dataset.foto_mes == 202006, "mextraccion_autoservicio"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccheques_depositados"] = None
    dataset.loc[dataset.foto_mes == 202006, "mcheques_depositados"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccheques_emitidos"] = None
    dataset.loc[dataset.foto_mes == 202006, "mcheques_emitidos"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccheques_depositados_rechazados"] = None
    dataset.loc[dataset.foto_mes == 202006, "mcheques_depositados_rechazados"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccheques_emitidos_rechazados"] = None
    dataset.loc[dataset.foto_mes == 202006, "mcheques_emitidos_rechazados"] = None
    dataset.loc[dataset.foto_mes == 202006, "tcallcenter"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccallcenter_transacciones"] = None
    dataset.loc[dataset.foto_mes == 202006, "thomebanking"] = None
    dataset.loc[dataset.foto_mes == 202006, "chomebanking_transacciones"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccajas_transacciones"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccajas_consultas"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccajas_depositos"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccajas_extracciones"] = None
    dataset.loc[dataset.foto_mes == 202006, "ccajas_otras"] = None
    dataset.loc[dataset.foto_mes == 202006, "catm_trx"] = None
    dataset.loc[dataset.foto_mes == 202006, "matm"] = None
    dataset.loc[dataset.foto_mes == 202006, "catm_trx_other"] = None
    dataset.loc[dataset.foto_mes == 202006, "matm_other"] = None
    dataset.loc[dataset.foto_mes == 202006, "ctrx_quarter"] = None
    dataset.loc[dataset.foto_mes == 202006, "cmobile_app_trx"] = None



# Aqui empieza el programa
OUTPUT['PARAM'] = PARAM
OUTPUT['time'] = {}
OUTPUT['time']['start'] = pd.Timestamp.now().strftime('%Y%m%d %H%M%S')

os.chdir(PARAM['home'])

# cargo el dataset
dataset = pd.read_csv(PARAM['dataset'], compression='gzip', low_memory=False)

# tmobile_app se daño a partir de 202010
dataset.drop(columns=['tmobile_app'], inplace=True)


# creo la carpeta donde va el experimento
os.makedirs(f"./exp/{PARAM['experimento']}/", exist_ok=True)
# Establezco el Working Directory DEL EXPERIMENTO
os.chdir(f"./exp/{PARAM['experimento']}/")

GrabarOutput()
with open('parametros.yml', 'w') as f:
    yaml.dump(PARAM, f) # escribo parametros utilizados

dataset.sort_values(by=['numero_de_cliente', 'foto_mes'], inplace=True)

# corrijo los  < foto_mes, campo >  que fueron pisados con cero
if PARAM['metodo'] == 'MachineLearning':
    Corregir_MachineLearning(dataset)
elif PARAM['metodo'] == 'EstadisticaClasica':
    Corregir_EstadisticaClasica(dataset)
else:
    print("No se aplica ninguna correccion.\n")


# grabo el dataset
dataset.to_csv('dataset.csv.gz', compression='gzip', index=False)

# guardo los campos que tiene el dataset
tb_campos = pd.DataFrame({
    'pos': range(1, dataset.shape[1]+1),
    'campo': dataset.columns,
    'tipo': dataset.dtypes,
    'nulos': dataset.isna().sum(),
    'ceros': (dataset == 0).sum()
})

tb_campos.to_csv('dataset.campos.txt', sep='\t', index=False)

OUTPUT['dataset']['ncol'] = dataset.shape[1]
OUTPUT['dataset']['nrow'] = dataset.shape[0]
OUTPUT['time']['end'] = pd.Timestamp.now().strftime('%Y%m%d %H%M%S')
GrabarOutput()

# dejo la marca final
with open('zRend.txt', 'a') as f:
    f.write(pd.Timestamp.now().strftime('%Y%m%d %H%M%S') + '\n')

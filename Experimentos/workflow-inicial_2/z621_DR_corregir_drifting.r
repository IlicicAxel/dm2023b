# Experimentos Colaborativos Default
# Workflow  Data Drifting repair

# limpio la memoria
rm(list = ls(all.names = TRUE)) # remove all objects
gc(full = TRUE) # garbage collection

require("data.table")
require("yaml")


# Parametros del script
PARAM <- list()
PARAM$experimento <- "DR6210_2"

PARAM$exp_input <- "CA6110_2"

PARAM$variables_intrames <- TRUE # atencion esto esta en TRUE

# valores posibles
#  "ninguno", "rank_simple", "rank_cero_fijo", "deflacion", "estandarizar"
PARAM$metodo <- "rank_cero_fijo"

PARAM$home <- "~/buckets/b1/"
# FIN Parametros del script

OUTPUT <- list()

#------------------------------------------------------------------------------

options(error = function() {
  traceback(20)
  options(error = NULL)
  stop("exiting after script error")
})
#------------------------------------------------------------------------------

GrabarOutput <- function() {
  write_yaml(OUTPUT, file = "output.yml") # grabo output
}
#------------------------------------------------------------------------------
# Esta es la parte que los alumnos deben desplegar todo su ingenio
# Agregar aqui sus PROPIAS VARIABLES manuales

AgregarVariables_IntraMes <- function(dataset) {
  gc()
  # INICIO de la seccion donde se deben hacer cambios con variables nuevas

  # creo un ctr_quarter que tenga en cuenta cuando
  # los clientes hace 3 menos meses que estan
  dataset[, ctrx_quarter_normalizado := ctrx_quarter]
  dataset[cliente_antiguedad == 1, ctrx_quarter_normalizado := ctrx_quarter * 5]
  dataset[cliente_antiguedad == 2, ctrx_quarter_normalizado := ctrx_quarter * 2]
  dataset[
    cliente_antiguedad == 3,
    ctrx_quarter_normalizado := ctrx_quarter * 1.2
  ]

  # variable extraida de una tesis de maestria de Irlanda
  dataset[, mpayroll_sobre_edad := mpayroll / cliente_edad]

  # se crean los nuevos campos para MasterCard  y Visa,
  #  teniendo en cuenta los NA's
  # varias formas de combinar Visa_status y Master_status
  dataset[, vm_status01 := pmax(Master_status, Visa_status, na.rm = TRUE)]
  dataset[, vm_status02 := Master_status + Visa_status]

  dataset[, vm_status03 := pmax(
    ifelse(is.na(Master_status), 10, Master_status),
    ifelse(is.na(Visa_status), 10, Visa_status)
  )]

  dataset[, vm_status04 := ifelse(is.na(Master_status), 10, Master_status)
    + ifelse(is.na(Visa_status), 10, Visa_status)]

  dataset[, vm_status05 := ifelse(is.na(Master_status), 10, Master_status)
    + 100 * ifelse(is.na(Visa_status), 10, Visa_status)]

  dataset[, vm_status06 := ifelse(is.na(Visa_status),
    ifelse(is.na(Master_status), 10, Master_status),
    Visa_status
  )]

  dataset[, mv_status07 := ifelse(is.na(Master_status),
    ifelse(is.na(Visa_status), 10, Visa_status),
    Master_status
  )]


  # combino MasterCard y Visa
  dataset[, vm_mfinanciacion_limite := rowSums(cbind(Master_mfinanciacion_limite, Visa_mfinanciacion_limite), na.rm = TRUE)]

  dataset[, vm_Fvencimiento := pmin(Master_Fvencimiento, Visa_Fvencimiento, na.rm = TRUE)]
  dataset[, vm_Finiciomora := pmin(Master_Finiciomora, Visa_Finiciomora, na.rm = TRUE)]
  dataset[, vm_msaldototal := rowSums(cbind(Master_msaldototal, Visa_msaldototal), na.rm = TRUE)]
  dataset[, vm_msaldopesos := rowSums(cbind(Master_msaldopesos, Visa_msaldopesos), na.rm = TRUE)]
  dataset[, vm_msaldodolares := rowSums(cbind(Master_msaldodolares, Visa_msaldodolares), na.rm = TRUE)]
  dataset[, vm_mconsumospesos := rowSums(cbind(Master_mconsumospesos, Visa_mconsumospesos), na.rm = TRUE)]
  dataset[, vm_mconsumosdolares := rowSums(cbind(Master_mconsumosdolares, Visa_mconsumosdolares), na.rm = TRUE)]
  dataset[, vm_mlimitecompra := rowSums(cbind(Master_mlimitecompra, Visa_mlimitecompra), na.rm = TRUE)]
  dataset[, vm_madelantopesos := rowSums(cbind(Master_madelantopesos, Visa_madelantopesos), na.rm = TRUE)]
  dataset[, vm_madelantodolares := rowSums(cbind(Master_madelantodolares, Visa_madelantodolares), na.rm = TRUE)]
  dataset[, vm_fultimo_cierre := pmax(Master_fultimo_cierre, Visa_fultimo_cierre, na.rm = TRUE)]
  dataset[, vm_mpagado := rowSums(cbind(Master_mpagado, Visa_mpagado), na.rm = TRUE)]
  dataset[, vm_mpagospesos := rowSums(cbind(Master_mpagospesos, Visa_mpagospesos), na.rm = TRUE)]
  dataset[, vm_mpagosdolares := rowSums(cbind(Master_mpagosdolares, Visa_mpagosdolares), na.rm = TRUE)]
  dataset[, vm_fechaalta := pmax(Master_fechaalta, Visa_fechaalta, na.rm = TRUE)]
  dataset[, vm_mconsumototal := rowSums(cbind(Master_mconsumototal, Visa_mconsumototal), na.rm = TRUE)]
  dataset[, vm_cconsumos := rowSums(cbind(Master_cconsumos, Visa_cconsumos), na.rm = TRUE)]
  dataset[, vm_cadelantosefectivo := rowSums(cbind(Master_cadelantosefectivo, Visa_cadelantosefectivo), na.rm = TRUE)]
  dataset[, vm_mpagominimo := rowSums(cbind(Master_mpagominimo, Visa_mpagominimo), na.rm = TRUE)]

  # a partir de aqui juego con la suma de Mastercard y Visa
  dataset[, vmr_Master_mlimitecompra := Master_mlimitecompra / vm_mlimitecompra]
  dataset[, vmr_Visa_mlimitecompra := Visa_mlimitecompra / vm_mlimitecompra]
  dataset[, vmr_msaldototal := vm_msaldototal / vm_mlimitecompra]
  dataset[, vmr_msaldopesos := vm_msaldopesos / vm_mlimitecompra]
  dataset[, vmr_msaldopesos2 := vm_msaldopesos / vm_msaldototal]
  dataset[, vmr_msaldodolares := vm_msaldodolares / vm_mlimitecompra]
  dataset[, vmr_msaldodolares2 := vm_msaldodolares / vm_msaldototal]
  dataset[, vmr_mconsumospesos := vm_mconsumospesos / vm_mlimitecompra]
  dataset[, vmr_mconsumosdolares := vm_mconsumosdolares / vm_mlimitecompra]
  dataset[, vmr_madelantopesos := vm_madelantopesos / vm_mlimitecompra]
  dataset[, vmr_madelantodolares := vm_madelantodolares / vm_mlimitecompra]
  dataset[, vmr_mpagado := vm_mpagado / vm_mlimitecompra]
  dataset[, vmr_mpagospesos := vm_mpagospesos / vm_mlimitecompra]
  dataset[, vmr_mpagosdolares := vm_mpagosdolares / vm_mlimitecompra]
  dataset[, vmr_mconsumototal := vm_mconsumototal / vm_mlimitecompra]
  dataset[, vmr_mpagominimo := vm_mpagominimo / vm_mlimitecompra]

  # Aqui debe usted agregar sus propias nuevas variables

  ## Ganancias banco
  dataset[, mmargen := rowSums(.SD, na.rm = TRUE), .SDcols = c("mactivos_margen", "mpasivos_margen")]
  dataset[, mmargen_x_producto := mmargen / cproductos]

  ## Total Pasivos
  dataset[, total_deuda := rowSums(.SD, na.rm = TRUE), .SDcols = c("mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios", "Visa_msaldototal", "Master_msaldototal")]

  ## Total Activos
  dataset[, total_activos := rowSums(.SD, na.rm = TRUE), .SDcols = c("mplazo_fijo_dolares", "mplazo_fijo_pesos", "minversion1_pesos", "minversion1_dolares", "minversion2", "mcuentas_saldo")]

  ## Balance en el banco
  dataset[, balance := total_activos - total_deuda]
  dataset[, ratio_deuda := total_deuda / (total_activos + 1)]

  ## saldos
  dataset[, has_cuentacorriente_saldo_pos := ifelse(mcuenta_corriente > 0, 1, 0) ]
  dataset[, has_cajaahorro_saldo_pos := ifelse(mcaja_ahorro > 0, 1, 0) ]
  dataset[, has_saldo_pos := ifelse(mcaja_ahorro + mcuenta_corriente > 0, 1, 0) ]

  ## Tiene movimientos/tarjetas
  dataset[, has_debito_transacciones := ifelse(dataset$ctarjeta_debito_transacciones > 0, 1, 0) ]
  dataset[, has_visa := ifelse(dataset$ctarjeta_visa > 0, 1, 0) ]
  dataset[, has_visa_transacciones := ifelse(dataset$ctarjeta_visa_transacciones > 0, 1, 0) ]
  dataset[, has_master := ifelse(dataset$ctarjeta_master > 0, 1, 0) ]
  dataset[, has_master_transacciones := ifelse(dataset$ctarjeta_master_transacciones > 0, 1, 0) ]

  ## Cantidad de tarjetas total
  dataset[, ctarjetas := rowSums(.SD, na.rm = TRUE), .SDcols = c("ctarjeta_visa", "ctarjeta_master")]

  ## Total seguros
  dataset[, cseguros := cseguro_vida + cseguro_auto + cseguro_vivienda + cseguro_accidentes_personales]

  ## Recibo pago de sueldo?
  dataset[, has_payroll := ifelse(dataset$cpayroll_trx + dataset$cpayroll2_trx  > 0, 1, 0) ]

  ## Total payroll
  dataset[, mpayroll_total := mpayroll + mpayroll2]

  ## Tiene débitos automáticos?
  dataset[, has_da := ifelse(dataset$ccuenta_debitos_automaticos + dataset$ctarjeta_visa_debitos_automaticos + dataset$ctarjeta_master_debitos_automaticos  > 0, 1, 0) ]

  ## cant pago mis cuentas?
  dataset[, has_pmc := ifelse(dataset$cpagomiscuentas  > 0, 1, 0) ]

  ## Total débitos automáticos
  dataset[, debitos_automaticos := mcuenta_debitos_automaticos + mttarjeta_visa_debitos_automaticos + mttarjeta_master_debitos_automaticos]

  ## Total Consumos y gastos
  dataset[, total_consumos := rowSums(.SD, na.rm = TRUE), .SDcols = c("mautoservicio", "mtarjeta_visa_consumo", "mtarjeta_master_consumo", "mpagodeservicios", "debitos_automaticos")]

  ## Total descuentos (sobre total de consumos?)
  dataset[, total_descuentos := rowSums(.SD, na.rm = TRUE), .SDcols = c("mtarjeta_visa_descuentos", "mtarjeta_master_descuentos", "mcajeros_propios_descuentos")]

  ## Descuentos sobre consumos
  dataset[, total_descuentos_sobre_consumos := ifelse(dataset$total_consumos == 0, 0, total_descuentos / total_consumos)]

  ## Total comisiones
  dataset[, has_comisiones := ifelse(dataset$ccomisiones_mantenimiento + dataset$ccomisiones_otras > 0, 1, 0) ]
  dataset[, total_comisiones := mcomisiones_mantenimiento + mcomisiones_otras]

  ## Balance transferencias
  dataset[, balance_transferencias := mtransferencias_recibidas - mtransferencias_emitidas]

  ## ¿hace más transacciones en cajeros de otros bancos?
  dataset[, cajeros_ajenos := ifelse(dataset$matm < dataset$matm_other, 1, 0)]

  ## ctrx quarter / cantidad de productos?
  dataset[, ctrx_x_producto := ctrx_quarter_normalizado / cproductos]

  ## comisiones / ctrx_quarter?
  dataset[, comisiones_x_trx := total_comisiones / (ctrx_quarter_normalizado + 1) ]

  # fechas tarjetas: llevo a años:
  dataset[, master_vencimiento := floor(dataset$Master_Fvencimiento/365)]
  dataset[, master_alta := floor(dataset$Master_fechaalta/365)]
  dataset[, visa_vencimiento := floor(dataset$Visa_Fvencimiento/365)]
  dataset[, visa_alta := floor(dataset$Visa_fechaalta/365)]

  ## limite de compra promedio
  dataset[, promedio_limite_compra := ifelse(dataset$ctarjetas == 0, 0, vm_mlimitecompra / ctarjetas) ]

  # pagado sobre saldo
  dataset[, pagado_sobre_saldo := ifelse(dataset$vm_msaldototal == 0, 0, vm_mpagado / vm_msaldototal) ]

  # consumo promedio
  dataset[, promedio_consumo := ifelse(dataset$vm_cconsumos == 0, 0, vm_mconsumototal /  vm_cconsumos) ]

  ## limite de compra sobre ingresos
  dataset[, limite_compra_sobre_ingresos := ifelse(dataset$mpayroll_total == 0, NA, vm_mlimitecompra / mpayroll_total) ]
  dataset[, limite_compra_sobre_activos := ifelse(dataset$total_activos == 0, NA, vm_mlimitecompra / total_activos) ]

  ## limite de compra real vs esperado según ingreso
  limite_esperado = median(dataset[mpayroll_total > 0, vm_mlimitecompra / mpayroll_total], na.rm=TRUE)
  dataset[, limite_compra_real_sobre_esperado := ifelse(dataset$total_activos == 0, NA, mpayroll_total * limite_esperado - vm_mlimitecompra) ]


  dataset[,"mmaster_consumo_transacciones_ratio"] = dataset[,"mtarjeta_master_consumo"] / dataset[,"ctarjeta_master_transacciones"]
  dataset[,"cmaster_descuentos_transacciones_ratio"] = dataset[,"ctarjeta_master_descuentos"] / (dataset[,"ctarjeta_master_transacciones"] + dataset[,"ctarjeta_master_debitos_automaticos"])
  dataset[,"mmaster_descuentos_transacciones_ratio"] = dataset[,"mtarjeta_master_descuentos"] / (dataset[,"mtarjeta_master_consumo"] + dataset[,"mttarjeta_master_debitos_automaticos"])
  dataset[,"mmaster_consumo_limite_ratio"] = dataset[,"mtarjeta_master_consumo"] / dataset[,"Master_mlimitecompra"]
  dataset[,"mmaster_consumo_limitef_ratio"] = dataset[,"mtarjeta_master_consumo"] / dataset[,"Master_mfinanciacion_limite"]

  dataset[,"mmaster_inicio_mora_s"] = dataset[,Master_Finiciomora < 180]
  dataset[,"mmaster_inicio_mora_a"] = dataset[,Master_Finiciomora < 360]
  dataset[,"mmaster_falta_s"] = dataset[,Master_fechaalta < 180]
  dataset[,"mmaster_falta_a"] = dataset[,Master_fechaalta < 360]
  dataset[,"mmaster_fvencimiento_q"] = dataset[,(Master_Fvencimiento > -90)]

  dataset[,"visa_vsaldo_limite"] = dataset[,"Visa_msaldototal"] / dataset[,"Visa_mlimitecompra"]
  dataset[,"visa_payroll_limite"] = dataset[,"mpayroll"] / dataset[,"Visa_mlimitecompra"]
  dataset[,"visa_saldo_limite"] = dataset[,"mcuentas_saldo"] / dataset[,"Visa_mlimitecompra"]

  dataset[,"visa_pagominimo_vsaldo"] = dataset[,"Visa_mpagominimo"] / dataset[,"Visa_msaldototal"]
  dataset[,"visa_pagominimo_limite"] = dataset[,"Visa_mpagominimo"] / dataset[,"Visa_mlimitecompra"]

  dataset[,"visa_adelanto_saldo"] = dataset[,"Visa_madelantopesos"] / dataset[,"Visa_msaldototal"]
  dataset[,"visa_adelanto_payroll"] = dataset[,"Visa_madelantopesos"] / dataset[,"mpayroll"]

  dataset[,"visa_payroll_saldo"] = dataset[,"mpayroll"] / dataset[,"Visa_msaldototal"]

  dataset[,"master_vsaldo_limite"] = dataset[,"Master_msaldototal"] / dataset[,"Master_mlimitecompra"]
  dataset[,"master_payroll_limite"] = dataset[,"mpayroll"] / dataset[,"Master_mlimitecompra"]
  dataset[,"master_saldo_limite"] = dataset[,"mcuentas_saldo"] / dataset[,"Master_mlimitecompra"]

  dataset[,"master_pagominimo_vsaldo"] = dataset[,"Master_mpagominimo"] / dataset[,"Master_msaldototal"]
  dataset[,"master_pagominimo_limite"] = dataset[,"Master_mpagominimo"] / dataset[,"Master_mlimitecompra"]

  dataset[,"master_adelanto_saldo"] = dataset[,"Master_madelantopesos"] / dataset[,"Master_msaldototal"]
  dataset[,"master_adelanto_payroll"] = dataset[,"Master_madelantopesos"] / dataset[,"mpayroll"]

  dataset[,"master_payroll_saldo"] = dataset[,"mpayroll"] / dataset[,"Master_msaldototal"]


  dataset[, "tenure_over_age"] = (dataset[, "cliente_antiguedad"] / 12) / dataset[, "cliente_edad"]


  dataset[, "master_vencimiento"] = dataset[, "Master_Fvencimiento"] / 365

  dataset[, "visa_vencimiento"] = dataset[, "Visa_Fvencimiento"] / 365

  dataset[, "master_alta"] = dataset[, "Master_fechaalta"] / 365

  dataset[, "visa_alta"] = dataset[, "Visa_fechaalta"] / 365

  # Variables by cliente_edad and cliente_antiguedad
  dataset[, "mpayroll_sobre_edad"] = dataset[, "mpayroll"] / dataset[, "cliente_edad"]
  dataset[, "mpayroll_sobre_antiguedad"] = dataset[, "mpayroll"] / dataset[, "cliente_antiguedad"]

  dataset[, "mpayroll2_sobre_edad"] = dataset[, "mpayroll2"] / dataset[, "cliente_edad"]
  dataset[, "mpayroll2_sobre_antiguedad"] = dataset[, "mpayroll2"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mfinanciacion_limite_sobre_edad"] = dataset[, "vm_mfinanciacion_limite"] / dataset[, "cliente_edad"]
  dataset[, "vm_mfinanciacion_limite_sobre_antiguedad"] = dataset[, "vm_mfinanciacion_limite"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_msaldototal_sobre_edad"] = dataset[, "vm_msaldototal"] / dataset[, "cliente_edad"]
  dataset[, "vm_msaldototal_sobre_antiguedad"] = dataset[, "vm_msaldototal"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_msaldopesos_sobre_edad"] = dataset[, "vm_msaldopesos"] / dataset[, "cliente_edad"]
  dataset[, "vm_msaldopesos_sobre_antiguedad"] = dataset[, "vm_msaldopesos"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_msaldodolares_sobre_edad"] = dataset[, "vm_msaldodolares"] / dataset[, "cliente_edad"]
  dataset[, "vm_msaldodolares_sobre_antiguedad"] = dataset[, "vm_msaldodolares"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mconsumospesos_sobre_edad"] = dataset[, "vm_mconsumospesos"] / dataset[, "cliente_edad"]
  dataset[, "vm_mconsumospesos_sobre_antiguedad"] = dataset[, "vm_mconsumospesos"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mconsumosdolares_sobre_edad"] = dataset[, "vm_mconsumosdolares"] / dataset[, "cliente_edad"]
  dataset[, "vm_mconsumosdolares_sobre_antiguedad"] = dataset[, "vm_mconsumosdolares"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mlimitecompra_sobre_edad"] = dataset[, "vm_mlimitecompra"] / dataset[, "cliente_edad"]
  dataset[, "vm_mlimitecompra_sobre_antiguedad"] = dataset[, "vm_mlimitecompra"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_madelantopesos_sobre_edad"] = dataset[, "vm_madelantopesos"] / dataset[, "cliente_edad"]
  dataset[, "vm_madelantopesos_sobre_antiguedad"] = dataset[, "vm_madelantopesos"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_madelantodolares_sobre_edad"] = dataset[, "vm_madelantodolares"] / dataset[, "cliente_edad"]
  dataset[, "vm_madelantodolares_sobre_antiguedad"] = dataset[, "vm_madelantodolares"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mpagado_sobre_edad"] = dataset[, "vm_mpagado"] / dataset[, "cliente_edad"]
  dataset[, "vm_mpagado_sobre_antiguedad"] = dataset[, "vm_mpagado"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mpagospesos_sobre_edad"] = dataset[, "vm_mpagospesos"] / dataset[, "cliente_edad"]
  dataset[, "vm_mpagospesos_sobre_antiguedad"] = dataset[, "vm_mpagospesos"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mpagosdolares_sobre_edad"] = dataset[, "vm_mpagosdolares"] / dataset[, "cliente_edad"]
  dataset[, "vm_mpagosdolares_sobre_antiguedad"] = dataset[, "vm_mpagosdolares"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mconsumototal_sobre_edad"] = dataset[, "vm_mconsumototal"] / dataset[, "cliente_edad"]
  dataset[, "vm_mconsumototal_sobre_antiguedad"] = dataset[, "vm_mconsumototal"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_cconsumos_sobre_edad"] = dataset[, "vm_cconsumos"] / dataset[, "cliente_edad"]
  dataset[, "vm_cconsumos_sobre_antiguedad"] = dataset[, "vm_cconsumos"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_cadelantosefectivo_sobre_edad"] = dataset[, "vm_cadelantosefectivo"] / dataset[, "cliente_edad"]
  dataset[, "vm_cadelantosefectivo_sobre_antiguedad"] = dataset[, "vm_cadelantosefectivo"] / dataset[, "cliente_antiguedad"]

  dataset[, "vm_mpagominimo_sobre_edad"] = dataset[, "vm_mpagominimo"] / dataset[, "cliente_edad"]
  dataset[, "vm_mpagominimo_sobre_antiguedad"] = dataset[, "vm_mpagominimo"] / dataset[, "cliente_antiguedad"]

  #ctrx_quarter_edad
  dataset[, "ctrx_quarter_edad"] = dataset[, "ctrx_quarter"] / dataset[, "cliente_edad"]
  dataset[, "ctrx_quarter_antiguedad"] = dataset[, "ctrx_quarter"] / dataset[, "cliente_antiguedad"]

  # Create variables calle  d *_risk 1 or 0 where:

# mpayroll < 10000
# cpayroll_trx = 0
# ctarjeta_visa_transacciones < 5
# ctarjeta_debito_transacciones < 2
# mcuentas_saldo = 0
# mcaja_ahorro < 10000
# mcuenta_corriente <= 0
# Mpasivos_margen <=0


# vm_financiacion_limite_risk < 175000
# vm_Finiciomora > 25
# vm_msaldototal < 12500
# total_consumo < 12500
# has_payroll = 0
# vm_consumopesos < 5000
# vm_limitecompra < 125000
# vm_fechaalta < 2200
# vm_cconsumos < 6
# vm_pagominimo = 0

  dataset[, "ctrx_quarter_risk"] = ifelse(dataset[, "ctrx_quarter"] < 50, 1, 0)
  dataset[, "ctrx_quarter_edad_risk"] = ifelse(dataset[, "ctrx_quarter_edad"] < 1, 1, 0)
  dataset[, "ctrx_quarter_antiguedad_risk"] = ifelse(dataset[, "ctrx_quarter_antiguedad"] < 0.5, 1, 0)
  dataset[, "mpayroll_risk"] = ifelse(dataset[, "mpayroll"] < 10000, 1, 0)
  dataset[, "cpayroll_trx_risk"] = ifelse(dataset[, "cpayroll_trx"] == 0, 1, 0)
  dataset[, "ctarjeta_visa_transacciones"] = ifelse(dataset[, "ctarjeta_visa_transacciones"] < 5, 1, 0)
  dataset[, "ctarjeta_debito_transacciones"] = ifelse(dataset[, "ctarjeta_debito_transacciones"] < 2, 1, 0)
  dataset[, "mcuentas_saldo_risk"] = ifelse(dataset[, "mcuentas_saldo"] == 0, 1, 0)
  dataset[, "mcaja_ahorro_risk"] = ifelse(dataset[, "mcaja_ahorro"] < 10000, 1, 0)
  dataset[, "mcuenta_corriente_risk"] = ifelse(dataset[, "mcuenta_corriente"] <= 0, 1, 0)
  dataset[, "mpasivos_margen_risk"] = ifelse(dataset[, "mpasivos_margen"] <= 0, 1, 0)
  dataset[, "vm_mfinanciacion_limite_risk"] = ifelse(dataset[, "vm_mfinanciacion_limite"] < 175000, 1, 0)
  dataset[, "vm_Finiciomora_risk"] = ifelse(dataset[, "vm_Finiciomora"] > 25, 1, 0)
  dataset[, "vm_msaldototal_risk"] = ifelse(dataset[, "vm_msaldototal"] < 12500, 1, 0)
  dataset[, "total_consumos_risk"] = ifelse(dataset[, "total_consumos"] < 12500, 1, 0)
  dataset[, "has_payroll_risk"] = ifelse(dataset[, "has_payroll"] == 0, 1, 0)
  dataset[, "vm_mconsumospesos_risk"] = ifelse(dataset[, "vm_mconsumospesos"] < 5000, 1, 0)
  dataset[, "vm_mlimitecompra_risk"] = ifelse(dataset[, "vm_mlimitecompra"] < 125000, 1, 0)
  dataset[, "vm_fechaalta_risk"] = ifelse(dataset[, "vm_fechaalta"] < 2200, 1, 0)
  dataset[, "vm_cconsumos_risk"] = ifelse(dataset[, "vm_cconsumos"] < 6, 1, 0)
  dataset[, "vm_mpagominimo_risk"] = ifelse(dataset[, "vm_mpagominimo"] == 0, 1, 0)
  #sum of all risks in variable total_risks
  dataset[, "total_risks"] = dataset[, "ctrx_quarter_risk"] + dataset[, "ctrx_quarter_edad_risk"] + dataset[, "ctrx_quarter_antiguedad_risk"] + dataset[, "mpayroll_risk"] + dataset[, "cpayroll_trx_risk"] + dataset[, "ctarjeta_visa_transacciones"] + dataset[, "ctarjeta_debito_transacciones"] + dataset[, "mcuentas_saldo_risk"] + dataset[, "mcaja_ahorro_risk"] + dataset[, "mcuenta_corriente_risk"] + dataset[, "mpasivos_margen_risk"] + dataset[, "vm_mfinanciacion_limite_risk"] + dataset[, "vm_Finiciomora_risk"] + dataset[, "vm_msaldototal_risk"] + dataset[, "total_consumos_risk"] + dataset[, "has_payroll_risk"] + dataset[, "vm_mconsumospesos_risk"] + dataset[, "vm_mlimitecompra_risk"] + dataset[, "vm_fechaalta_risk"] + dataset[, "vm_cconsumos_risk"] + dataset[, "vm_mpagominimo_risk"]
  
  # valvula de seguridad para evitar valores infinitos
  # paso los infinitos a NULOS
  infinitos <- lapply(
    names(dataset),
    function(.name) dataset[, sum(is.infinite(get(.name)))]
  )

  infinitos_qty <- sum(unlist(infinitos))
  if (infinitos_qty > 0) {
    cat(
      "ATENCION, hay", infinitos_qty,
      "valores infinitos en tu dataset. Seran pasados a NA\n"
    )
    dataset[mapply(is.infinite, dataset)] <<- NA
  }


  # valvula de seguridad para evitar valores NaN  que es 0/0
  # paso los NaN a 0 , decision polemica si las hay
  # se invita a asignar un valor razonable segun la semantica del campo creado
  nans <- lapply(
    names(dataset),
    function(.name) dataset[, sum(is.nan(get(.name)))]
  )

  nans_qty <- sum(unlist(nans))
  if (nans_qty > 0) {
    cat(
      "ATENCION, hay", nans_qty,
      "valores NaN 0/0 en tu dataset. Seran pasados arbitrariamente a 0\n"
    )

    cat("Si no te gusta la decision, modifica a gusto el programa!\n\n")
    dataset[mapply(is.nan, dataset)] <<- 0
  }
}
#------------------------------------------------------------------------------
# deflaciona por IPC
# momento 1.0  31-dic-2020 a las 23:59

drift_deflacion <- function(campos_monetarios) {
  vfoto_mes <- c(
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105, 202106,
    202107, 202108, 202109
  )

  vIPC <- c(
    1.9903030878, 1.9174403544, 1.8296186587,
    1.7728862972, 1.7212488323, 1.6776304408,
    1.6431248196, 1.5814483345, 1.4947526791,
    1.4484037589, 1.3913580777, 1.3404220402,
    1.3154288912, 1.2921698342, 1.2472681797,
    1.2300475145, 1.2118694724, 1.1881073259,
    1.1693969743, 1.1375456949, 1.1065619600,
    1.0681100000, 1.0370000000, 1.0000000000,
    0.9680542110, 0.9344152616, 0.8882274350,
    0.8532444140, 0.8251880213, 0.8003763543,
    0.7763107219, 0.7566381305, 0.7289384687
  )

  tb_IPC <- data.table(
    "foto_mes" = vfoto_mes,
    "IPC" = vIPC
  )

  dataset[tb_IPC,
    on = c("foto_mes"),
    (campos_monetarios) := .SD * i.IPC,
    .SDcols = campos_monetarios
  ]
}

#------------------------------------------------------------------------------

drift_rank_simple <- function(campos_drift) {
  for (campo in campos_drift)
  {
    cat(campo, " ")
    dataset[, paste0(campo, "_rank") :=
      (frank(get(campo), ties.method = "random") - 1) / (.N - 1), by = foto_mes]
    dataset[, (campo) := NULL]
  }
}
#------------------------------------------------------------------------------
# El cero se transforma en cero
# los positivos se rankean por su lado
# los negativos se rankean por su lado

drift_rank_cero_fijo <- function(campos_drift) {
  for (campo in campos_drift)
  {
    cat(campo, " ")
    dataset[get(campo) == 0, paste0(campo, "_rank") := 0]
    dataset[get(campo) > 0, paste0(campo, "_rank") :=
      frank(get(campo), ties.method = "random") / .N, by = foto_mes]

    dataset[get(campo) < 0, paste0(campo, "_rank") :=
      -frank(-get(campo), ties.method = "random") / .N, by = foto_mes]
    dataset[, (campo) := NULL]
  }
}
#------------------------------------------------------------------------------

drift_estandarizar <- function(campos_drift) {
  for (campo in campos_drift)
  {
    cat(campo, " ")
    dataset[, paste0(campo, "_normal") := 
      (get(campo) -mean(get(campo), na.rm=TRUE)) / sd(get(campo), na.rm=TRUE),
      by = foto_mes]

    dataset[, (campo) := NULL]
  }
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui comienza el programa
OUTPUT$PARAM <- PARAM
OUTPUT$time$start <- format(Sys.time(), "%Y%m%d %H%M%S")

setwd(PARAM$home)

# cargo el dataset donde voy a entrenar
# esta en la carpeta del exp_input y siempre se llama  dataset.csv.gz
dataset_input <- paste0("./exp/", PARAM$exp_input, "/dataset.csv.gz")
dataset <- fread(dataset_input)

# creo la carpeta donde va el experimento
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)
# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))

GrabarOutput()
write_yaml(PARAM, file = "parametros.yml") # escribo parametros utilizados

# primero agrego las variables manuales
if (PARAM$variables_intrames) AgregarVariables_IntraMes(dataset)

# ordeno de esta forma por el ranking
setorder(dataset, foto_mes, numero_de_cliente)

# por como armé los nombres de campos,
#  estos son los campos que expresan variables monetarias
campos_monetarios <- colnames(dataset)
campos_monetarios <- campos_monetarios[campos_monetarios %like%
  "^(m|Visa_m|Master_m|vm_m)"]

# aqui aplico un metodo para atacar el data drifting
# hay que probar experimentalmente cual funciona mejor
switch(PARAM$metodo,
  "ninguno"        = cat("No hay correccion del data drifting"),
  "rank_simple"    = drift_rank_simple(campos_monetarios),
  "rank_cero_fijo" = drift_rank_cero_fijo(campos_monetarios),
  "deflacion"      = drift_deflacion(campos_monetarios),
  "estandarizar"   = drift_estandarizar(campos_monetarios)
)



fwrite(dataset,
  file = "dataset.csv.gz",
  logical01 = TRUE,
  sep = ","
)

#------------------------------------------------------------------------------

# guardo los campos que tiene el dataset
tb_campos <- as.data.table(list(
  "pos" = 1:ncol(dataset),
  "campo" = names(sapply(dataset, class)),
  "tipo" = sapply(dataset, class),
  "nulos" = sapply(dataset, function(x) {
    sum(is.na(x))
  }),
  "ceros" = sapply(dataset, function(x) {
    sum(x == 0, na.rm = TRUE)
  })
))

fwrite(tb_campos,
  file = "dataset.campos.txt",
  sep = "\t"
)

#------------------------------------------------------------------------------
OUTPUT$dataset$ncol <- ncol(dataset)
OUTPUT$dataset$nrow <- nrow(dataset)
OUTPUT$time$end <- format(Sys.time(), "%Y%m%d %H%M%S")
GrabarOutput()

# dejo la marca final
cat(format(Sys.time(), "%Y%m%d %H%M%S"), "\n",
  file = "zRend.txt",
  append = TRUE
)

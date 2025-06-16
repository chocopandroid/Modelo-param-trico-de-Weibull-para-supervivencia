import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.subset import subset
from pandas.api.types import CategoricalDtype
# from scipy.stats import Normal

datos_DATA = pd.read_csv("subpoblaciones_DATA.csv")
datos_DEMO = pd.read_csv("subpoblaciones_DEMO.csv")

datos_todos = pd.merge(datos_DATA, datos_DEMO, on='ID')

#datosss = pd.merge(datos_DATA, datos_DEMO, on='ID')
datosss = datos_DEMO
#hago una copia por si acaso, para no modificar datos iniciales
datos = datosss.copy()


# grafico de tumor segun sexo
conteo = datos.groupby(['SEXO', 'TUMOR']).size().reset_index(name='count')


conteo['label'] = conteo.apply(
    lambda row: f"{row['SEXO']} - {row['TUMOR']}\n({row['count']} pacientes)", axis=1
)

# Colores definidos por sexo
colors = {
    'Varon': '#90EE90',     # verde claro
    'Mujer': '#FFB6C1',     # rosa claro
}


conteo['color'] = conteo['SEXO'].map(colors)

# Gráfico de circular
plt.figure()
plt.pie(
    conteo['count'],
    labels=conteo['label'],
    autopct='%1.1f%%',
    startangle=90,
    colors=conteo['color'],
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Distribución inicial por sexo y tipo de tumor')
plt.tight_layout()



mapa_estado = {'FALLECIDO':1, 'VIVO':0}

datos['ESTADO'] = datos['ESTADO'].map(mapa_estado)

# Paso a variables categoricas:
sexo_cat = CategoricalDtype(categories=['Varon', 'Mujer'], ordered=False)
tumor_cat = CategoricalDtype(categories=['PULMON', 'MELANOMA'], ordered=False)
estadio_cat = CategoricalDtype(categories=['II', 'III', 'IV'], ordered=True)
respuesta_cat = CategoricalDtype(categories=['RC', 'RP', 'EE', 'PD'], ordered=True)
edad_cat = CategoricalDtype(ordered=True)

datos['SEXO'] = datos['SEXO'].astype(sexo_cat)
datos['TUMOR'] = datos['TUMOR'].astype(tumor_cat)
datos['ESTADIO'] = datos['ESTADIO'].astype(estadio_cat)
datos['RESPUESTA'] = datos['RESPUESTA'].astype(respuesta_cat)

datos['SEXO_CODE'] = datos['SEXO'].cat.codes
datos['TUMOR_CODE'] = datos['TUMOR'].cat.codes
datos['ESTADIO_CODE'] = datos['ESTADIO'].cat.codes
datos['RESPUESTA_CODE'] = datos['RESPUESTA'].cat.codes
datos['EDAD_CODE'] = datos['EDAD'] if pd.api.types.is_numeric_dtype(datos['EDAD']) else datos['EDAD'].cat.codes


# Datos y => Supervivencia en meses y estado (fallecido/vivo)
# Calculo tiempo de supervivencia en meses y vivo/muerto al final de inmuno

datos['FECHA ÚLTIMO SEGUIMIENTO'] = pd.to_datetime(datos['FECHA ÚLTIMO SEGUIMIENTO'])
datos['INICIO'] = pd.to_datetime(datos['INICIO'])
datos['surv meses'] = (datos['FECHA ÚLTIMO SEGUIMIENTO'].dt.year - datos['INICIO'].dt.year)*12.0 + (datos['FECHA ÚLTIMO SEGUIMIENTO'].dt.month - datos['INICIO'].dt.month)

datos[['ID','SEXO', 'SEXO_CODE', 'EDAD','EDAD_CODE','TUMOR', 'TUMOR_CODE','ESTADIO','ESTADIO_CODE', 'RESPUESTA','RESPUESTA_CODE', 'surv meses']] = datos[['ID', 'SEXO', 'SEXO_CODE', 'EDAD','EDAD_CODE','TUMOR', 'TUMOR_CODE','ESTADIO','ESTADIO_CODE', 'RESPUESTA','RESPUESTA_CODE', 'surv meses']].dropna()

datos_usar = datos[['ID', 'SEXO', 'SEXO_CODE', 'EDAD','EDAD_CODE','TUMOR', 'TUMOR_CODE','ESTADIO','ESTADIO_CODE', 'RESPUESTA','RESPUESTA_CODE']]
y = datos[['ID', 'ESTADO', 'surv meses']].copy()


datos_totales = pd.merge(datos_usar, y)

y['ESTADO'] = y['ESTADO'].astype(bool)


# Imputacion de datos faltantes:
### solo elimino los que tienen todos los datos faltantes
datos_usar = datos_usar.dropna(axis=0)
y = y.dropna(axis=0)
datos_totales = datos_totales.dropna(axis=0)

# Establezco los modelos de estimacion, etc
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter, WeibullAFTFitter
from lifelines.utils import k_fold_cross_validation


# Contar combinaciones de SEXO y TUMOR
conteo = datos_totales.groupby(['SEXO', 'TUMOR']).size().reset_index(name='count')

# Crear etiquetas sin errores: mostrar n=por combinación SEXO-TUMOR
conteo['label'] = conteo.apply(
    lambda row: f"{row['SEXO']} - {row['TUMOR']}\n({row['count']} pacientes)", axis=1
)

# Colores definidos por sexo (se asignan al conjunto completo)
colors = {
    'Varon': '#90EE90',
    'Mujer': '#FFB6C1',
}

# Asignar color según sexo
conteo['color'] = conteo['SEXO'].map(colors)

# Gráfico de pastel
plt.figure()
plt.pie(
    conteo['count'],
    labels=conteo['label'],
    autopct='%1.1f%%',
    startangle=90,
    colors=conteo['color'],
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Distribución por sexo y tipo de tumor tras preprocesado')
plt.tight_layout()


### Kaplan-Meier:
kmf = KaplanMeierFitter()

kmf.fit(y['surv meses'], event_observed=y['ESTADO'])
plt.figure()
kmf.plot()
plt.title('Curva Kaplan-Meier - Estado')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')


# Hago la curva separando por tipo de tumor(pulmon, melanoma), respuesta, estadio
## --> 4 curvas
# tumor:
plt.figure()
for tipo_tumor in datos_totales['TUMOR'].cat.categories:
    subset = datos_totales[datos_totales['TUMOR'] == tipo_tumor]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_tumor)
    kmf.plot()

plt.title('Curva Kaplan-Meier - Tipo de tumor')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()


# respuesta:
plt.figure()
for tipo_respuesta in datos['RESPUESTA'].cat.categories:
    subset = datos[datos['RESPUESTA'] == tipo_respuesta]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_respuesta)
    kmf.plot()

plt.title('Curva Kaplan-Meier - Respuesta')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()


# estadio:
plt.figure()
for tipo_estadio in datos['ESTADIO'].cat.categories:
    subset = datos[datos['ESTADIO'] == tipo_estadio]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_estadio)
    kmf.plot()

plt.title('Curva Kaplan-Meier en función del estadio')
plt.xlabel('tiempo de superv. en meses')
plt.ylabel('prob de supervivencia')
plt.legend()


# sexo:
plt.figure()
for tipo_sexo in datos['SEXO'].cat.categories:
    subset = datos[datos['SEXO'] == tipo_sexo]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_sexo)
    kmf.plot()

plt.title('Curva Kaplan-Meier - Sexo')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()


### CoxPH
# hay grupos con muy pocas personas = se distorsiona el modelo
# --> agrupo los valores que tienen muy pocos pacientes en el siguiente

datos['ESTADIO_GRUPO'] = datos['ESTADIO'].replace({
    'II': 'II U III',
    'III': 'II U III',
    'IV': 'IV',
})

estadio_grupo_cat = CategoricalDtype(categories=['II U III', 'IV'], ordered=True)

datos['ESTADIO_GRUPO'] = datos['ESTADIO_GRUPO'].astype(estadio_grupo_cat)

datos['ESTADIO_GRUPO_CODE'] = datos['ESTADIO_GRUPO'].cat.codes

variables_dummies = pd.get_dummies(datos[['SEXO', 'TUMOR', 'ESTADIO_GRUPO', 'RESPUESTA']], drop_first=True)


## datos MODELO 1
datos_cox = pd.concat([datos[['ID']], variables_dummies], axis=1)
#datos_cox = datos[['ID','EDAD','SEXO', 'TUMOR', 'ESTADIO_GRUPO', 'RESPUESTA']]
datos_totales_cox = pd.merge(datos_cox, y, on='ID')
datos_totales_cox = datos_totales_cox.drop(columns=['ID'])
datos_totales_cox['ESTADO'] = datos_totales_cox['ESTADO'].astype(bool)

datos_code = datos[['SEXO_CODE', 'TUMOR_CODE','ESTADIO_GRUPO_CODE','RESPUESTA_CODE', 'surv meses', 'ESTADO']].dropna()

from lifelines.utils import concordance_index

# ## MODELO 1 (todas las vars.): CoxPHFitter
cph = CoxPHFitter()
# cox_model = datos_code.rename(columns={
#     'ESTADIO_GRUPO_CODE': 'ESTADIO',
#     'RESPUESTA_CODE': 'RESPUESTA',
#     'TUMOR_CODE': 'TUMOR',
#     'SEXO_CODE': 'SEXO',
#
#
# })

cph2 = cph.fit(datos_code, duration_col='surv meses', event_col='ESTADO')
print(cph2.summary)
plt.figure()
cph2.plot()
plt.yticks(fontsize=10)
plt.title('Regresión de Cox - Datos demográficos')
plt.subplots_adjust(left=0.3)


plt.figure()
cph2.plot_partial_effects_on_outcome('SEXO_CODE', values=[0,1])
plt.title("Efecto parcial del sexo sobre la supervivencia \n(modelo Cox - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(["Varón", "Mujer", "baseline"])

plt.figure()
cph2.plot_partial_effects_on_outcome('RESPUESTA_CODE', values=[0,1,2,3])
plt.title("Efecto parcial de la respuesta sobre la supervivencia \n(modelo Cox - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['RC', 'RP', 'EE', 'PD','baseline'])

plt.figure()
cph2.plot_partial_effects_on_outcome(covariates='TUMOR_CODE', values=[0,1])
plt.title("Efecto parcial del tumor sobre la supervivencia \n(modelo Cox - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(["Pulmón","Melanoma", "baseline"])


plt.figure()
cph2.plot_partial_effects_on_outcome(covariates='ESTADIO_GRUPO_CODE', values=[0,1])
plt.title("Efecto parcial del estadio sobre la supervivencia \n(modelo Cox - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['II U III', 'IV', 'baseline'])

# #sexo, tumor, estadio
plt.figure()
cph2.plot_partial_effects_on_outcome(covariates=['SEXO_CODE','TUMOR_CODE','ESTADIO_GRUPO_CODE'],
                                      values=[[0,0,0],[0,0,1],[1,0,0],[1,0,1],[0,1,0],[0,1,1],[1,1,0],[1,1,1]])
plt.title("Efecto parcial del sexo, tumor, y estadio \n(modelo Cox - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['Hombre, Pulmón, Estadio II U III','Hombre, Pulmón, Estadio IV',
            'Mujer, Pulmón, Estadio II U III', 'Mujer, Pulmón, Estadio IV',
            'Hombre, Melanoma, Estadio II U III', 'Hombre, Melanoma, Estadio IV',
            'Mujer, Melanoma, Estadio II U III', 'Mujer, Melanoma, Estadio IV'], fontsize='small')

#tumor, respuesta
plt.figure()
cph2.plot_partial_effects_on_outcome(covariates=['RESPUESTA_CODE', 'TUMOR_CODE'],
                                     values=[[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1]])
plt.title("Efecto parcial de la respuesta y tumor sobre la supervivencia \n(modelo Cox - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['RC, Pulmón', 'RP, Pulmón', 'EE, Pulmón', 'PD, Pulmón',
            'RC, Melanoma', 'RP, Melanoma', 'EE, Melanoma', 'PD, Melanoma'], fontsize='small')



# MOSTRAR CATEGORÍAS DE REFERENCIA
print("\nCategorías de referencia (omitidas en el modelo):")
print("RESPUESTA: RC (comparado contra RP, EE, PD)")
print("TUMOR: PULMON (comparado contra MELANOMA)")
print("SEXO: Varón (comparado contra Mujer)")
print("ESTADIO_GRUPO: II U III (comparado contra IV)")

plt.figure()
base = cph.baseline_cumulative_hazard_.plot(drawstyle='steps')
base.set_title("Baseline Cumulative Hazard (Modelo de Cox - Datos demográficos)")
base.set_xlabel("Tiempo")
base.set_ylabel("Riesgo acumulado")
base.legend(["Riesgo acumulado basal"])
plt.title('Baseline Cumulative Hazard')

cph.print_summary()

### Weibull
wbf = WeibullFitter()
weibul = wbf.fit(datos_totales['surv meses'], event_observed=datos_totales['ESTADO'], label='Weibull') # tengo en cuenta los datos censurados

plt.figure()
weibul.plot_survival_function()
plt.title('Curva de Supervivencia - Weibull')
plt.xlabel('Tiempo (meses)')
plt.ylabel('Probabilidad de Supervivencia')
print(f'Lambda de Weibull: {wbf.lambda_}')
print(wbf.summary)

plt.figure()
wbf.plot_cumulative_hazard()
plt.title("Riesgo acumulado (Weibull - Datos demográficos)")
plt.legend()

mediana = wbf.median_survival_time_
n = len(datos_totales)
predicciones = np.full(n, -mediana)
duracion=datos_totales['surv meses']
evento=datos_totales['ESTADO']

c_index = concordance_index(duracion, predicciones, evento)

kmf.fit(datos_totales['surv meses'], datos_totales['ESTADO'], label='Kaplan-Meier')

# comparacion km - Weibul
plt.figure()
kmf.plot_survival_function()
weibul.plot_survival_function()

plt.title("Comparación: Kaplan-Meier vs. Weibull (Datos demográficos) ")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend()
plt.show()

## intento el weibulAFTfitter
aft = WeibullAFTFitter()
columnas_aft = ['SEXO_CODE', 'TUMOR_CODE', 'ESTADIO_CODE', 'RESPUESTA_CODE', 'surv meses', 'ESTADO']

datos_aft = datos_totales[columnas_aft] #datos en CategoricalDtypes
# aft_model = datos_aft.rename(columns={
#     'ESTADIO_CODE': 'ESTADIO',
#     'RESPUESTA_CODE': 'RESPUESTA',
#     'TUMOR_CODE': 'TUMOR',
#     'SEXO_CODE': 'SEXO',
#     # 'NLR_GRUPO_CODE': 'NLR'
#     # Agrega más si es necesario
# })
aft.fit(datos_aft, duration_col='surv meses', event_col='ESTADO', ancillary=True)
# uso los datos convertidos en dummies para poder ver todas las etiquetas
#aft.fit(datos_totales_cox, duration_col='surv meses', event_col='ESTADO', ancillary=True)

plt.figure()
plt.yticks(fontsize=10)
plt.title(r'Modelado de $\rho$ y $\lambda$ - Weibull AFT')
aft.plot()
plt.subplots_adjust(left=0.3)

plt.figure()
plt.title(r'Modelado de $\rho$ y $\lambda$ - Weibull AFT (catdtypes)')
aft2.plot()


# Mostrar cómo afectan las distintas variables a la curva de supervivencia:
# --> plot_partial_effects_on_outcome()

#sexo
plt.figure()
aft.plot_partial_effects_on_outcome('SEXO_CODE', values=[0,1])
plt.title("Efecto parcial del sexo sobre la supervivencia \n(modelo Weibull AFT- Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(["Varón", "Mujer", "baseline"])

#tumor
plt.figure()
aft.plot_partial_effects_on_outcome(covariates='TUMOR_CODE', values=[0,1])
plt.title("Efecto parcial del tumor sobre la supervivencia \n(modelo Weibull AFT - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(["Pulmón","Melanoma", "baseline"])

respuesta
plt.figure()
aft.plot_partial_effects_on_outcome(covariates='RESPUESTA_CODE', values=[0,1,2,3])
plt.title("Efecto parcial de la respuesta sobre la supervivencia \n(modelo Weibull AFT - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['RC', 'RP', 'EE', 'PD','baseline'])

#estadio
plt.figure()
aft.plot_partial_effects_on_outcome(covariates='ESTADIO_CODE', values=[0,1])
plt.title("Efecto parcial del estadio sobre la supervivencia \n(modelo Weibull AFT - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['II U III', 'IV', 'baseline'])

#sexo, tumor, estadio
plt.figure()
aft.plot_partial_effects_on_outcome(covariates=['SEXO_CODE','TUMOR_CODE','ESTADIO_CODE'],
                                      values=[[0,0,0],[0,0,1],[1,0,0],[1,0,1],[0,1,0],[0,1,1],[1,1,0],[1,1,1]])
plt.title("Efecto parcial del sexo,tumor y estadio sobre la supervivencia \n(modelo Weibull AFT - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['Hombre, Pulmón, Estadio II U III','Hombre, Pulmón, Estadio IV',
            'Mujer, Pulmón, Estadio II U III', 'Mujer, Pulmón, Estadio IV',
            'Hombre, Melanoma, Estadio II U III', 'Hombre, Melanoma, Estadio IV',
            'Mujer, Melanoma, Estadio II U III', 'Mujer, Melanoma, Estadio IV'])


plt.figure()
aft.plot_partial_effects_on_outcome(covariates=['RESPUESTA_CODE', 'TUMOR_CODE'],
                                     values=[[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1]])
plt.title("Efecto parcial de la respuesta y tumor sobre la supervivencia \n(modelo Weibull AFT - Datos demográficos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['RC, Pulmón', 'RP, Pulmón', 'EE, Pulmón', 'PD, Pulmón',
            'RC, Melanoma', 'RP, Melanoma', 'EE, Melanoma', 'PD, Melanoma'])


###################################################################
## Evaluacion de los modelos
print(f"Concordance index DEMO (Cox): {cph2.concordance_index_}")
print(f"Concordance index (Cox - Dummies): {cph.concordance_index_}")
print(f"Concordance index (Weibull): {c_index}")
print(f"Concordance index DEMO (Weibull AFT): {aft.concordance_index_}")

print(f"Log-likelihood DEMO:"
      f"\nCox: {cph.log_likelihood_}"
      f"\nWeibull: {weibul.log_likelihood_}"
      f"\nWeibull AFT: {aft.log_likelihood_}")

print(f"Log-likelihood ratio test (Cox): {cph2.log_likelihood_ratio_test()}")

print(f"AIC DEMO \n(Weibull AFT): {aft.AIC_}"
      f"\n AIC (Weibull): {wbf.AIC_}"
      f"\nAIC Parcial (Cox): {cph2.AIC_partial_}")


from sklearn.metrics import brier_score_loss
t = 24 #tiempo de evaluacion en meses
cox_preds = cph2.predict_survival_function(datos_code, times=[t]).T.squeeze()
weibul_preds = wbf.survival_function_at_times(t).values.repeat(len(datos_totales))
aft_preds = aft.predict_survival_function(datos_aft, times=[t]).T.squeeze()

evento_cox = (datos_totales_cox['surv meses']<=t)&(datos_totales_cox['ESTADO']==1)
evento_wbf = (datos_totales['surv meses']<=t)&(datos_totales['ESTADO']==1)
evento_aft = (datos_aft['surv meses']<=t)&(datos_aft['ESTADO']==1)


y_cox = (datos_totales_cox['surv meses']>t).astype(int)
y_cox = (datos_code['surv meses']>t).astype(int)
y_wbf = (datos_totales['surv meses']>t).astype(int)
y_aft = (datos_aft['surv meses']>t).astype(int)

brier_cox = brier_score_loss(y_cox, cox_preds)
brier_weibul = brier_score_loss(y_wbf, weibul_preds)
brier_aft = brier_score_loss(y_aft, aft_preds)

print(f"Brier score - MODELO DEMO:"
      f"\nCox: {brier_cox:.4f}"
      f"\nWeibull: {brier_weibul:.4f}"
      f"\nWeibull AFT: {brier_aft:.4f}")


# K-FOLD
scores_cox = k_fold_cross_validation(cph,datos_totales_cox,'surv meses',
                                 event_col='ESTADO',k=5,scoring_method="concordance_index")

scores_aft = k_fold_cross_validation(aft,datos_aft,'surv meses',
                                 event_col='ESTADO',k=5,scoring_method="concordance_index")

print(f"Resultado k-fold DEMO \nCOX:  {scores_cox}"
      f"\n Weibull AFT {scores_aft}")

media_aft = np.mean(scores_aft)
desviacion_aft = np.std(scores_aft)

media_cox = np.mean(scores_cox)
desviacion = np.std(scores_cox)

print(f"C-index medio AFT: {media_aft:.4f}"
      f"\nC-index medio COX: {media_cox:.4f}")
print(f"Desviación estándar AFT: {desviacion_aft:.4f}"
      f"\nDesviación estándar COX: {desviacion:.4f}")


aic_cph = cph2.AIC_partial_       # Solo log-parcial, típico de modelos de Cox
aic_wbf = weibul.AIC_
aic_aft = aft.AIC_

aics = np.array([aic_wbf, aic_aft])
delta_aic = aics - np.min(aics)
akaike_weights = np.exp(-0.5 * delta_aic) / np.sum(np.exp(-0.5 * delta_aic))

# 5. Mostrar resultados
resultados = pd.DataFrame({
    'Modelo': [ 'WeibullFitter', 'WeibullAFTFitter'],
    'AIC': aics,
    'Delta AIC': delta_aic,
    'Peso de Akaike': akaike_weights
})

print(resultados.sort_values('Peso de Akaike', ascending=False))

plt.show()





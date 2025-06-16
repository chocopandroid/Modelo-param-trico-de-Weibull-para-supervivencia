import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype


datos_DATA = pd.read_csv("subpoblaciones_DATA.csv")
datos_DEMO = pd.read_csv("subpoblaciones_DEMO.csv")

datos_todos = pd.merge(datos_DEMO, datos_DATA, on='ID')

datos_mes0 = datos_todos[datos_todos['month']==0].copy()

# Paso a variables categoricas:
sexo_cat = CategoricalDtype(categories=['Varon', 'Mujer'], ordered=False)
tumor_cat = CategoricalDtype(categories=['PULMON', 'MELANOMA'], ordered=False)
estadio_cat = CategoricalDtype(categories=['II', 'III', 'IV'], ordered=True)
respuesta_cat = CategoricalDtype(categories=['RC', 'RP', 'EE', 'PD'], ordered=True)
edad_cat = CategoricalDtype(ordered=True)

nlr_cat = CategoricalDtype(categories=['<5', '≥5'], ordered=True)
nlr_cat3 =CategoricalDtype(categories=['<3.53', '≥3.53'], ordered=True)
nlr_grupos_cat = CategoricalDtype(categories=['<1', '1-2', '2-3', '3-4', '4-5', '5-6','≥6'], ordered=True)
def categories_nlr(x):
    if x < 1:
        return '<1'
    elif x < 2:
        return '1-2'
    elif x < 3:
        return '2-3'
    elif x < 4:
        return '3-4'
    elif x < 5:
        return '4-5'
    elif x < 6:
        return '5-6'
    else:
        return '≥6'

datos_mes0['SEXO'] = datos_mes0['SEXO'].astype(sexo_cat)
datos_mes0['TUMOR'] = datos_mes0['TUMOR'].astype(tumor_cat)
datos_mes0['ESTADIO'] = datos_mes0['ESTADIO'].astype(estadio_cat)
datos_mes0['RESPUESTA'] = datos_mes0['RESPUESTA'].astype(respuesta_cat)
datos_mes0['NLR_GRUPO'] = datos_mes0['NLR'].apply(lambda x: '<5' if x < 5 else '≥5')
datos_mes0['NLR_GRUPO3'] = datos_mes0['NLR'].apply(lambda x: '<3.53' if x < 3.53 else '≥3.53')
datos_mes0['NLR_GRUPOSS'] =datos_mes0['NLR'].apply(categories_nlr)

datos_mes0['SEXO_CODE'] = datos_mes0['SEXO'].cat.codes
datos_mes0['TUMOR_CODE'] = datos_mes0['TUMOR'].cat.codes
datos_mes0['ESTADIO_CODE'] = datos_mes0['ESTADIO'].cat.codes
datos_mes0['RESPUESTA_CODE'] = datos_mes0['RESPUESTA'].cat.codes
datos_mes0['EDAD_CODE'] = datos_mes0['EDAD'] if pd.api.types.is_numeric_dtype(datos_mes0['EDAD']) \
                                            else datos_mes0['EDAD'].cat.codes


datos_mes0['NLR_GRUPO'] = datos_mes0['NLR_GRUPO'].astype(nlr_cat)
datos_mes0['NLR_GRUPO3'] = datos_mes0['NLR_GRUPO3'].astype(nlr_cat3)
datos_mes0['NLR_GRUPOSS'] =datos_mes0['NLR_GRUPOSS'].astype(nlr_grupos_cat)

datos_mes0['NLR_GRUPO_CODE'] = datos_mes0['NLR_GRUPO'].cat.codes
datos_mes0['NLR_GRUPO_CODE3'] = datos_mes0['NLR_GRUPO3'].cat.codes
datos_mes0['NLR_GRUPOSS_CODE'] = datos_mes0['NLR_GRUPOSS'].cat.codes

datos_mes0['FECHA ÚLTIMO SEGUIMIENTO'] = pd.to_datetime(datos_mes0['FECHA ÚLTIMO SEGUIMIENTO'])
datos_mes0['INICIO'] = pd.to_datetime(datos_mes0['INICIO'])
datos_mes0['surv meses'] = ((datos_mes0['FECHA ÚLTIMO SEGUIMIENTO'].dt.year - datos_mes0['INICIO'].dt.year)*12.0 +
                            (datos_mes0['FECHA ÚLTIMO SEGUIMIENTO'].dt.month - datos_mes0['INICIO'].dt.month))

datos_mes0 = datos_mes0.dropna(subset=['ID', 'SEXO', 'SEXO_CODE', 'EDAD','EDAD_CODE','TUMOR', 'TUMOR_CODE','ESTADIO',
                                       'ESTADIO_CODE', 'RESPUESTA','RESPUESTA_CODE', 'surv meses', 'NLR_GRUPO',
                                       'NLR_GRUPO_CODE','NLR_GRUPOSS','NLR_GRUPOSS_CODE'])

datos_usar_mes0 = datos_mes0[['ID', 'SEXO', 'SEXO_CODE', 'EDAD','TUMOR', 'TUMOR_CODE','ESTADIO','ESTADIO_CODE',
                              'RESPUESTA','RESPUESTA_CODE', 'NEUTROPHIL', 'LYMPHOCYTE', 'NLR', 'NLR_GRUPO',
                              'NLR_GRUPO_CODE', 'NLR_GRUPO3']].copy()
y0 = datos_mes0[['ID', 'ESTADO', 'surv meses']].copy()
y0['ESTADO'] = y0['ESTADO'].astype(bool)

datos_totales0 = pd.merge(datos_usar_mes0, y0, on='ID')

datos_bio = datos_mes0[['ID','NLR']].copy().dropna()

datos_bio_SOLOS = pd.merge(datos_bio, y0, on='ID')
datos_bio_SOLOS = datos_bio_SOLOS.dropna()

datos_totales0 = datos_totales0.dropna()

# Establezco los modelos de estimacion, etc
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter, WeibullAFTFitter
from lifelines.utils import k_fold_cross_validation

datos_totales0['ESTADIO_GRUPO'] = datos_totales0['ESTADIO'].replace({
    'II': 'II U III',
    'III': 'II U III',
    'IV': 'IV',
})
print(datos_totales0['ESTADIO_GRUPO'].value_counts(dropna=False))

estadio_grupo_cat = CategoricalDtype(categories=['II U III', 'IV'], ordered=True)

datos_totales0['ESTADIO_GRUPO'] = datos_totales0['ESTADIO_GRUPO'].astype(estadio_grupo_cat)

datos_totales0['ESTADIO_GRUPO_CODE'] = datos_totales0['ESTADIO_GRUPO'].cat.codes

datos_totales0 = datos_totales0.dropna()

### KAPLAN-MEIER: NO SE PUEDE USAR CON VARS. CONTINUAS
kmf = KaplanMeierFitter()
plt.figure()
for tipo_estadio in datos_totales0['NLR_GRUPO'].cat.categories:
    subset = datos_totales0[datos_totales0['NLR_GRUPO'] == tipo_estadio]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_estadio)
    kmf.plot()

plt.title('Curva Kaplan-Meier - NLR < 5 (mes 0)')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()

plt.figure()
for tipo_estadio in datos_totales0['NLR_GRUPO3'].cat.categories:
    subset = datos_totales0[datos_totales0['NLR_GRUPO3'] == tipo_estadio]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_estadio)
    kmf.plot()

plt.title('Curva Kaplan-Meier - NLR < 3.53 (mes 0)')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()

plt.figure()
for tipo_estadio in datos_totales0['SEXO'].cat.categories:
    subset = datos_totales0[datos_totales0['SEXO'] == tipo_estadio]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_estadio)
    kmf.plot()

plt.title('Curva Kaplan-Meier - Sexo (mes 0)')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()

plt.figure()
for tipo_estadio in datos_totales0['ESTADIO'].cat.categories:
    subset = datos_totales0[datos_totales0['ESTADIO'] == tipo_estadio]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_estadio)
    kmf.plot()

plt.title('Curva Kaplan-Meier - Estadio (mes 0)')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()

plt.figure()
for tipo_estadio in datos_totales0['RESPUESTA'].cat.categories:
    subset = datos_totales0[datos_totales0['RESPUESTA'] == tipo_estadio]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_estadio)
    kmf.plot()

plt.title('Curva Kaplan-Meier - Respuesta (mes 0)')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()

plt.figure()
for tipo_estadio in datos_totales0['TUMOR'].cat.categories:
    subset = datos_totales0[datos_totales0['TUMOR'] == tipo_estadio]

    kmf.fit(subset['surv meses'], event_observed=subset['ESTADO'], label=tipo_estadio)
    kmf.plot()

plt.title('Curva Kaplan-Meier - Tumor (mes 0)')
plt.xlabel('Tiempo de supervivencia (meses)')
plt.ylabel('Probabilidad de supervivencia')
plt.legend()


### COX
cph = CoxPHFitter()
variables_dummies0 = pd.get_dummies(datos_totales0[['SEXO', 'TUMOR', 'ESTADIO_GRUPO', 'RESPUESTA']], drop_first=True)

datos_cox0 = pd.concat([datos_bio, variables_dummies0], axis=1)
datos_totales_cox0 = pd.merge(datos_cox0, y0, on='ID')

datos_totales_cox0 = datos_totales_cox0.dropna()
datos_totales_cox0 = datos_totales_cox0.drop(columns=['ID'])
datos_totales_cox0['ESTADO'] = datos_totales_cox0['ESTADO'].astype(bool)

datos_bio_SOLOS = datos_bio_SOLOS.drop(columns='ID')
datos_bio_SOLOS['ESTADO'] = datos_bio_SOLOS['ESTADO'].astype(bool)

# ## MODELO 1 (todas las vars.): CoxPHFitter
cph = CoxPHFitter()
cph2 = cph.fit(datos_bio_SOLOS, duration_col='surv meses', event_col='ESTADO')
print(cph2.summary)
plt.figure()
cph2.plot()
plt.title('Regresión de Cox - Biomarcadores')

plt.figure()
base = cph2.baseline_cumulative_hazard_.plot(drawstyle='steps')
base.set_title("Baseline Cumulative Hazard (Modelo de Cox - Biomarcadores)")
base.set_xlabel("Tiempo")
base.set_ylabel("Riesgo acumulado")
base.legend(["Riesgo acumulado basal"])

# ## MODELO 2 (todas las vars. + continuas): CoxPHFitter
cph3=CoxPHFitter()
cph3.fit(datos_totales_cox0, duration_col='surv meses', event_col='ESTADO')
print(cph3.summary)
plt.figure()
cph3.plot()
plt.yticks(fontsize=10)
plt.title('Regresión de Cox - Datos completos')
plt.subplots_adjust(left=0.3)

print(datos_totales_cox0.columns.tolist())

plt.figure()
base = cph3.baseline_cumulative_hazard_.plot(drawstyle='steps')
base.set_title("Baseline Cumulative Hazard (Modelo de Cox - Datos completos)")
base.set_xlabel("Tiempo")
base.set_ylabel("Riesgo acumulado")
base.legend(["Riesgo acumulado basal"])

datos_code = datos_totales0[['SEXO_CODE', 'TUMOR_CODE','ESTADIO_GRUPO_CODE','RESPUESTA_CODE', 'surv meses', 'ESTADO', 'NLR_GRUPO_CODE']].dropna()
# cox_model = datos_code.rename(columns={
#     'ESTADIO_GRUPO_CODE': 'ESTADIO',
#     'RESPUESTA_CODE': 'RESPUESTA',
#     'TUMOR_CODE': 'TUMOR',
#     'SEXO_CODE': 'SEXO',
# })

cph2 = cph.fit(datos_code, duration_col='surv meses', event_col='ESTADO')


plt.figure()
cph2.plot_partial_effects_on_outcome('SEXO_CODE', values=[0,1])
plt.title("Efecto parcial del sexo sobre la supervivencia \n(modelo Cox - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(["Varón", "Mujer", "baseline"])

plt.figure()
cph2.plot_partial_effects_on_outcome('RESPUESTA_CODE', values=[0,1,2,3])
plt.title("Efecto parcial de la respuesta sobre la supervivencia \n(modelo Cox - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['RC', 'RP', 'EE', 'PD','baseline'])

plt.figure()
cph2.plot_partial_effects_on_outcome(covariates='TUMOR_CODE', values=[0,1])
plt.title("Efecto parcial del tumor sobre la supervivencia \n(modelo Cox - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(["Pulmón","Melanoma", "baseline"])


plt.figure()
cph2.plot_partial_effects_on_outcome(covariates='ESTADIO_GRUPO_CODE', values=[0,1])
plt.title("Efecto parcial del estadio sobre la supervivencia \n(modelo Cox - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['II U III', 'IV', 'baseline'])

#sexo, tumor, estadio
plt.figure()
cph2.plot_partial_effects_on_outcome(covariates=['SEXO_CODE','TUMOR_CODE','ESTADIO_GRUPO_CODE'],
                                      values=[[0,0,0],[0,0,1],[1,0,0],[1,0,1],[0,1,0],[0,1,1],[1,1,0],[1,1,1]])
plt.title("Efecto parcial del sexo, tumor, y estadio \n(modelo Cox - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['Hombre, Pulmón, Estadio II U III','Hombre, Pulmón, Estadio IV',
            'Mujer, Pulmón, Estadio II U III', 'Mujer, Pulmón, Estadio IV',
            'Hombre, Melanoma, Estadio II U III', 'Hombre, Melanoma, Estadio IV',
            'Mujer, Melanoma, Estadio II U III', 'Mujer, Melanoma, Estadio IV'])

#tumor, respuesta
plt.figure()
cph2.plot_partial_effects_on_outcome(covariates=['RESPUESTA_CODE', 'TUMOR_CODE'],
                                     values=[[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1]])
plt.title("Efecto parcial de la respuesta y tumor sobre la supervivencia \n(modelo Cox - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['RC, Pulmón', 'RP, Pulmón', 'EE, Pulmón', 'PD, Pulmón',
            'RC, Melanoma', 'RP, Melanoma', 'EE, Melanoma', 'PD, Melanoma'])




### WEIBULL:
wbf = WeibullFitter()

weibul2 = wbf.fit(datos_totales0['surv meses'], event_observed=datos_totales0['ESTADO']) # tengo en cuenta los datos censurados
#
# plt.figure()
# weibul2.plot_survival_function()
# plt.title('Curva de Supervivencia - Weibull (Datos completos)')
# plt.xlabel('Tiempo (meses)')
# plt.ylabel('Probabilidad de Supervivencia')
# print(f'Lambda de Weibull COMPLETOS: {weibul2.lambda_}')
#
weibul3 = WeibullFitter()
weibul3 = wbf.fit(datos_bio_SOLOS['surv meses'], event_observed=datos_bio_SOLOS['ESTADO']) # tengo en cuenta los datos censurados

plt.figure()
weibul3.plot_survival_function()
plt.title('Curva de Supervivencia - Weibull (Biomarcadores)')
plt.xlabel('Tiempo (meses)')
plt.ylabel('Probabilidad de Supervivencia')
print(f'Lambda de Weibull BIOMADS: {weibul3.lambda_}')

from lifelines.utils import concordance_index

mediana = wbf.median_survival_time_
n = len(datos_totales0)
predicciones = np.full(n, -mediana)
duracion=datos_totales0['surv meses']
evento=datos_totales0['ESTADO']

c_index = concordance_index(duracion, predicciones, evento)

print(c_index)

print(weibul2.summary)

plt.figure()
wbf.plot_cumulative_hazard()
plt.title("Riesgo acumulado (Weibull - Datos completos)")
plt.legend()
plt.show()

# Ajustar modelos - Comparacion KM y Weibull
kmf.plot_survival_function()
weibul2.plot_survival_function()

plt.title("Comparación: Kaplan-Meier vs. Weibull (Datos completos) ")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend()
plt.show()

### WEIBULL AFT
aft = WeibullAFTFitter()


aft2 = aft.fit(datos_bio_SOLOS, duration_col='surv meses', event_col='ESTADO')
plt.figure()
aft2.plot()
plt.yticks(fontsize=10)
plt.title(r'Modelado de $\rho$ y $\lambda$ - Weibull AFT (Biomarcadores)')
plt.subplots_adjust(left=0.3)


columnas_aft0=['SEXO', 'TUMOR', 'ESTADIO_GRUPO', 'RESPUESTA', 'NLR_GRUPO', 'surv meses', 'ESTADO']
datos_aft20 = datos_totales0[columnas_aft0] #datos en CategoricalDtype
datos_aft20 = datos_aft20.dropna()
# Renombrar columnas para graficar
# aft_model = datos_aft20.rename(columns={
#     'ESTADIO_CODE': 'ESTADIO',
#     'RESPUESTA_CODE': 'RESPUESTA',
#     'TUMOR_CODE': 'TUMOR',
#     'SEXO_CODE': 'SEXO',
#     'NLR_GRUPO_CODE': 'NLR'
#     # Agrega más si es necesario
# })
aft20 = aft.fit(datos_aft20, duration_col='surv meses', event_col='ESTADO', ancillary=True, formula='C(SEXO) + C(TUMOR) + C(ESTADIO_GRUPO) + C(RESPUESTA) +C(NLR_GRUPO)')

plt.figure()
aft20.plot()
plt.yticks(fontsize=10)
plt.title(r'Modelado de $\rho$ y $\lambda$ - Weibull AFT (Datos completos)')
plt.subplots_adjust(left=0.3)


# Mostrar cómo afectan las distintas variables a la curva de supervivencia:
# --> plot_partial_effects_on_outcome()
# sexo
plt.figure()
aft20.plot_partial_effects_on_outcome('SEXO_CODE', values=[0,1])
plt.title("Efecto parcial del sexo sobre la supervivencia \n(modelo Weibull AFT - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(["Varón", "Mujer", "baseline"])


#tumor
plt.figure()
aft20.plot_partial_effects_on_outcome(covariates='TUMOR_CODE', values=[0,1])
plt.title("Efecto parcial del tumor sobre la supervivencia \n(modelo Weibull AFT - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(["Pulmón","Melanoma", "baseline"])

respuesta
plt.figure()
aft20.plot_partial_effects_on_outcome(covariates=['RESPUESTA_CODE'],
                                     values=[0,1,2,3])
plt.title("Efecto parcial de la respuesta sobre la supervivencia \n(modelo Weibull AFT- Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['RC', 'RP', 'EE', 'PD'])

#estadio
plt.figure()
aft20.plot_partial_effects_on_outcome(covariates='ESTADIO_CODE', values=[0,1])
plt.title("Efecto parcial del estadio sobre la supervivencia \n(modelo Weibull AFT - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['II U III', 'IV', 'baseline'])

#sexo, tumor, estadio
plt.figure()
aft20.plot_partial_effects_on_outcome(covariates=['SEXO_CODE','TUMOR_CODE','ESTADIO_CODE'],
                                      values=[[0,0,0],[0,0,1],[1,0,0],[1,0,1],[0,1,0],[0,1,1],[1,1,0],[1,1,1]])
plt.title("Efecto parcial del sexo, tumor, y estadio \n(modelo Weibull AFT - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['Hombre, Pulmón, Estadio II U III','Hombre, Pulmón, Estadio IV',
            'Mujer, Pulmón, Estadio II U III', 'Mujer, Pulmón, Estadio IV',
            'Hombre, Melanoma, Estadio II U III', 'Hombre, Melanoma, Estadio IV',
            'Mujer, Melanoma, Estadio II U III', 'Mujer, Melanoma, Estadio IV'])

tumor, respuesta
plt.figure()
aft20.plot_partial_effects_on_outcome(covariates=['RESPUESTA_CODE', 'TUMOR_CODE'],
                                     values=[[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1]])
plt.title("Efecto parcial de la respuesta y tumor sobre la supervivencia \n(modelo Weibull AFT- Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['RC, Pulmón', 'RP, Pulmón', 'EE, Pulmón', 'PD, Pulmón',
            'RC, Melanoma', 'RP, Melanoma', 'EE, Melanoma', 'PD, Melanoma'])



plt.figure()
aft20.plot_partial_effects_on_outcome(covariates='NLR_GRUPO', values=['<5', '≥5'])
plt.title("Efecto parcial de NLR sobre supervivencia \n(modelo Weibull AFT - Datos completos)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['NLR<5','NLR≥5'])

datos_aft_b = datos_mes0[['NLR_GRUPOSS', 'ESTADO','surv meses']].copy()
datos_aft_b['ESTADO']=datos_aft_b['ESTADO'].astype(bool)

aft_new = datos_aft_b.rename(columns={
    'NLR_GRUPOSS': 'NLR_GRUPOS',
})

aft_b = WeibullAFTFitter()

aft_b.fit(aft_new, duration_col='surv meses', event_col='ESTADO', formula='C(NLR_GRUPOS)')
plt.figure()
aft_b.plot()
plt.yticks(fontsize=10)
plt.title('Modelado de ' + r'$\rho$' + ' y ' + r'$\lambda$' + ' - Weibull AFT\n(Biomarcadores - NLR dividido)', fontsize=12)
plt.subplots_adjust(left=0.3)



plt.figure()
aft_b.plot_partial_effects_on_outcome(covariates='NLR_GRUPOS', values=['<1', '1-2', '2-3', '3-4', '4-5','5-6', '≥6'])
plt.title("Efecto parcial de NLR sobre supervivencia \n(modelo Weibull AFT - Biomarcadores con NLR dividido)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['<1','1-2','2-3','3-4','4-5','5-6','≥6'])


datos_mes0['ESTADIO_GRUPO'] = datos_mes0['ESTADIO'].replace({
    'II': 'II U III',
    'III': 'II U III',
    'IV': 'IV',
})
datos_mes0['ESTADIO_GRUPO_CODE'] = datos_mes0['ESTADIO_GRUPO'].cat.codes
# datos_code_aft = datos_mes0[['SEXO_CODE', 'TUMOR_CODE','ESTADIO_GRUPO_CODE','RESPUESTA_CODE', 'surv meses', 'ESTADO', 'NLR_GRUPOSS']].copy().dropna()
datos_code_aft = datos_mes0[['SEXO', 'TUMOR','ESTADIO_GRUPO','RESPUESTA', 'surv meses', 'ESTADO', 'NLR_GRUPOSS']].copy().dropna()

datos_code_aft['ESTADO'] =datos_code_aft['ESTADO'].astype(bool)
aft_code_new = datos_code_aft.rename(columns={
    'NLR_GRUPOSS': 'NLR_GRUPOS',
})
aft_div_todos = WeibullAFTFitter()
aft_div_todos.fit(aft_code_new, duration_col='surv meses', event_col='ESTADO', formula='C(SEXO) + C(TUMOR) + C(ESTADIO_GRUPO) + C(RESPUESTA) +C(NLR_GRUPOS)')


plt.figure()
aft_div_todos.plot_partial_effects_on_outcome(covariates='NLR_GRUPOS', values=['<1', '1-2', '2-3', '3-4', '4-5','5-6', '≥6'])
plt.title("Efecto parcial de NLR sobre supervivencia \n(modelo Weibull AFT - Datos completos con NLR dividido)")
plt.xlabel("Tiempo (meses)")
plt.ylabel("Probabilidad de supervivencia")
plt.legend(['<1','1-2','2-3','3-4','4-5', '5-6','≥6'])


## Evaluacion de los modelos
print(f"Concordance index (Cox): {cph2.concordance_index_}")
# print(f"Concordance index (Weibull): {c_index}")
print(f"Concordance index (Weibull AFT): {aft20.concordance_index_}")

print(f"Log-likelihood:"
      f"\nCox: {cph.log_likelihood_}"
      f"\nWeibull: {weibul2.log_likelihood_}"
      f"\nWeibull AFT: {aft20.log_likelihood_}")

print(f"Log-likelihood ratio test (Cox): {cph.log_likelihood_ratio_test()}")

from sklearn.metrics import brier_score_loss

t = 24 #tiempo de evaluacion en meses
cox_preds = cph2.predict_survival_function(datos_code, times=[t]).T.squeeze()
weibul_preds = weibul2.survival_function_at_times(t).values.repeat(len(datos_totales0))
aft_preds = aft20.predict_survival_function(datos_aft20, times=[t]).T.squeeze()

evento_cox = (datos_code['surv meses']<=t)&(datos_code['ESTADO']==1)
evento_wbf = (datos_totales0['surv meses']<=t)&(datos_totales0['ESTADO']==1)
evento_aft = (datos_code['surv meses']<=t)&(datos_code['ESTADO']==1)

y_cox = (datos_code['surv meses']>t).astype(int)
y_wbf = (datos_totales0['surv meses']>t).astype(int)
y_aft = (datos_code['surv meses']>t).astype(int)

brier_cox = brier_score_loss(y_cox, cox_preds)
brier_weibul = brier_score_loss(y_wbf, weibul_preds)
brier_aft = brier_score_loss(y_aft, aft_preds)

print(f"Brier score - MODELO 1:"
      f"\nCox: {brier_cox:.4f}"
      f"\nWeibull: {brier_weibul:.4f}"
      f"\nWeibull AFT: {brier_aft:.4f}")


# K-FOLD
scores_cox = k_fold_cross_validation(cph2,datos_code,'surv meses',
                                 event_col='ESTADO',k=7,scoring_method="concordance_index")
# scores_wbf = k_fold_cross_validation(wbf,datos_totales,'surv meses',
#                                  event_col='ESTADO',k=5,scoring_method="concordance_index")

aft_model = WeibullAFTFitter()
scores_aft = k_fold_cross_validation(aft_model,datos_aft20,'surv meses',
                                 event_col='ESTADO',k=7,scoring_method="concordance_index",fitter_kwargs={"formula":'C(SEXO) + C(TUMOR) + C(ESTADIO_GRUPO) + C(RESPUESTA) + C(NLR_GRUPO)'}
)

print(f"Resultado k-fold TODOS \nCOX:  {scores_cox}"
      f"\n Weibull AFT {scores_aft}")
media_aft = np.mean(scores_aft)
desviacion_aft = np.std(scores_aft)

media_cox = np.mean(scores_cox)
desviacion = np.std(scores_cox)

print(f"C-index medio AFT: {media_aft:.4f}"
      f"\nC-index medio COX: {media_cox:.4f}")
print(f"Desviación estándar AFT: {desviacion_aft:.4f}"
      f"\nDesviación estándar COX: {desviacion:.4f}")


aic_cph = cph2.AIC_partial_
aic_wbf = weibul2.AIC_
aic_aft = aft20.AIC_


print(f"AIC (Weibull AFT): {aft20.AIC_}"
      f"\nCox: {cph2.AIC_partial_}"
      f"\nWeibull: {weibul2.AIC_}")
print(f"AIC (Weibull AFT): {aic_aft}"
      f"\nCox: {aic_cph}"
      f"\nWeibull: {aic_wbf}")


aics = np.array([ aic_wbf, aic_aft])
delta_aic = aics - np.min(aics)
akaike_weights = np.exp(-0.5 * delta_aic) / np.sum(np.exp(-0.5 * delta_aic))

resultados = pd.DataFrame({
    'Modelo': [ 'WeibullFitter', 'WeibullAFTFitter'],
    'AIC': aics,
    'Delta AIC': delta_aic,
    'Peso de Akaike': akaike_weights
})

print(resultados.sort_values('Peso de Akaike', ascending=False))

aics2 = np.array([aic_cph, aic_wbf, aic_aft])
delta_aic2 = aics2 - np.min(aics2)
akaike_weights2 = np.exp(-0.5 * delta_aic2) / np.sum(np.exp(-0.5 * delta_aic2))

resultados2 = pd.DataFrame({
    'Modelo': [ 'CoxPHFitter','WeibullFitter', 'WeibullAFTFitter'],
    'AIC': aics2,
    'Delta AIC': delta_aic2,
    'Peso de Akaike': akaike_weights2
})

print(resultados2.sort_values('Peso de Akaike', ascending=False))


plt.show()
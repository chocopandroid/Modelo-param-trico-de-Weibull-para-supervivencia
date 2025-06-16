#### REG. MULTINOMIAL PARA PREDECIR RESPUESTA EN FUNCIÓN DE PARAMETROS:
from importlib import import_module

# 0=RC, 1=RP, 2=EE, 3=PD
##DATOS: LINFO T CD3          76
#        LINFO T CD4          76
#        LINFO T CD8          76
#        LINFO B              76
#        LINFO NK             76
#        LINFO CD45           76
#        COCIENTE CD4/CD8     76
#        NEUTROPHIL          177
#        LYMPHOCYTE          177
#        NLR                 177
#        LDH                 125

# columnas proteina s100B, CEA, cyfra 21.1, SCC y CA 15-3 no se van a usar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, std, absolute
from numpy.ma.core import absolute
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
#from sklearn.preprocessing import LabelEncoder --> en orden alfabetico, tambien valida

datos_DATA = pd.read_csv("subpoblaciones_DATA.csv")
datos_DEMO = pd.read_csv("subpoblaciones_DEMO.csv")

datos = pd.merge(datos_DATA, datos_DEMO, on='ID')

print(datos_DATA['month'].value_counts())

# Paso la respuesta a valores numéricos
mapa_respuestas = {'RC':0, 'RP':1, 'EE':2, 'PD':3}
respuestas_posibles = ['RC', 'RP', 'EE', 'PD']
respuestas_codificadas = [mapa_respuestas[x] for x in respuestas_posibles]

# Añado columna nueva para las respuestas codificadas
#datos_limpios['respuesta codificada'] = datos_limpios['RESPUESTA'].map(mapa_respuestas)
datos['respuesta codificada'] = datos['RESPUESTA'].map(mapa_respuestas)

# Selecciono datos solo del mes 0
datos_mes0 = datos[datos['month'] == 0]
# quitamos los sujetos a los que le falta LINFO CD4:
datos_mes0 = datos_mes0.loc[datos_mes0["respuesta codificada"].isna()==False]
# datos_mes0 = datos_mes0.loc[datos_mes0["LINFO T CD4"].isna()==False]
#meses = datos.iloc[:, 1]
#print(datos_mes0.head())

# Elimino valores que faltan (missing values)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#1: eliminar columnas donde falte >30% de los datos
missing_cols = [col for col in datos_mes0.columns
                if datos_mes0[col].isnull().sum() > len(datos_mes0)/1.5]
#COJO EL 66,66% porque sino 4 variables a tratar me las quito de golpe
filtro = datos_mes0.select_dtypes(exclude=['object'])
cols_drop = [col for col in (missing_cols) if col in filtro.columns]
filtro = filtro.drop(columns=cols_drop)
# porque tienen 110 missing values
#para usar MICE hay q quitar los valores categoricos:

#2: imputar datos con MICE
imputer = IterativeImputer(random_state=0, max_iter=5, sample_posterior=True)
datos_imp = pd.DataFrame(imputer.fit_transform(filtro), columns=filtro.columns)
# datos_imp = filtro.copy()

#3: antes de MICE, eliminar filas con datos faltantes y computar
# ==> comparar con dataset MICE y ver si difieren los modelos

# Establezco el modelo
variables = [el for el in filtro.columns if el in ['LINFO T CD3', 'LINFO T CD4', 'LINFO T CD8', 'LINFO B', 'LINFO NK', 'LINFO CD45', 'COCIENTE CD4/CD8', 'NEUTROPHIL', 'LYMPHOCYTE', 'NLR', 'LDH']]
print(f"Variables a utilizar: {variables}")
#n_samples = 50
X = datos_imp[variables]
y = datos_imp['respuesta codificada'].astype(int)
#print(y.head(15))
# fig, axis = plt.subplots(ncols=int(np.ceil(np.sqrt(len(variables)))),
#                          nrows=int(np.ceil(np.sqrt(len(variables)))))
# for i, ax in enumerate(axis.flatten()):
#     if i>=len(variables):
#         break
#     ax.scatter(X.iloc[:,i], y)
#     ax.set_xlabel(variables[i])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Escalado de datos 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold


escalar = StandardScaler()
X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test)

cv = KFold(n_splits=2, random_state=1, shuffle=True)
# for train, test in kf.split(X):
#     print("%s %s" % (train, test))


from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.base import BaseEstimator, clone
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
##model = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=1000)
#model = svm.LinearSVC(class_weight="balanced") #, C=0.01)
#model = tree.DecisionTreeClassifier()
#model = ensemble.RandomForestClassifier()

#clf = ensemble.RandomForestClassifier()
#clf = svm.LinearSVC(class_weight="balanced")
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
class OrdinalClassifier(BaseEstimator):
    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                try:
                  clf.module
                except: # For others
                  clf.fit(X, binary_y)
                else: # For MLP
                  binary_y_reshape = binary_y.astype('float32').reshape(-1,1)
                  clf.fit(X, binary_y_reshape)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[i][:,1])
            elif i in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[i-1][:,1] - clfs_predict[i][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[i-1][:,1])
        try:
          self.clf.module
        except: # For others
          pred_proba = np.vstack(predicted).T
        else: # For MLP
            pred_proba = np.hstack((predicted))

        return pred_proba

    def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y, sample_weight=None):
            _, indexed_y = np.unique(y, return_inverse=True)
            return accuracy_score(indexed_y, self.predict(X), sample_weight=sample_weight)


ord_mod = OrdinalClassifier(clf)

model.fit(X_train, y_train)
ord_mod.fit(X_train, y_train)

# predicciones:
#y_pred = model.predict(X_test)
y_pred = ord_mod.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)
acc = accuracy_score(y_test, y_pred)
print("accuracy:", acc)
scores = cross_val_score(estimator = model, X = X_train, y = y_train, cv = cv)
print("cross val. score:", scores)
kfold = mean(absolute(scores))
print("kfold:", kfold)


ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=respuestas_posibles,
    normalize="true",
)
plt.show()

# Decodificacion de las respuestas para la prediccion
#inverso = {x: k for k, x in mapa_respuestas.items()}
#respuestas_decodificadas = [inverso[n] for n in respuestas_codificadas]
#datos['respuesta decodificada'] = datos['respuesta codificada'].map(mapa_respuestas)





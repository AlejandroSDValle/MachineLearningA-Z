#Importar la s librerias
import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd;

#importar el dataset
dataset = pd.read_csv('50_Startups.csv');
X = dataset.iloc[:, :-1].values;
Y = dataset.iloc[:, 4].values;

#Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder;
labelencoder_X = LabelEncoder();
X[:,3] = labelencoder_X.fit_transform(X[:,3]);
onehotencoder = OneHotEncoder(categorical_features=[3]);
X = onehotencoder.fit_transform(X).toarray();

#Eliminar una de las variables Ficticias
X = X[:, 1:];

#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.cross_validation import train_test_split;
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0);

from sklearn.linear_model import LinearRegression
regression = LinearRegression();
regression.fit(X_train, Y_train);

#prediccion de los resultados en el conjunto de testing
y_pred = regression.predict(X_test);

#construir el modelo optimo de la RLM utilizando la Eliminacion hacia atras
import statsmodels.api as sm;
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1);
SL = 0.05;

X_opt = X[:, [0,1,2,3,4,5]];
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit();
datos = regression_OLS.summary();

X_opt = X[:, [0,1,3,4,5]];
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit();
datos = regression_OLS.summary();

X_opt = X[:, [0,3,4,5]];
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit();
datos = regression_OLS.summary();

X_opt = X[:, [0,3,5]];
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit();
datos = regression_OLS.summary();

X_opt = X[:, [0,3]];
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit();
datos = regression_OLS.summary();











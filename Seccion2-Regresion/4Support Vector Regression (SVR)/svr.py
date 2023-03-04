#Importar librerias
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values;
Y = dataset.iloc[:,2].values;

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
sc_y = StandardScaler();
X = sc_X.fit_transform(X);
Y = sc_y.fit_transform(Y.reshape(-1,1));

#ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression();
lin_reg.fit(X,Y);

#Ajustar la regresion polinomica con el dataset
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X,Y)

#Prediccion de nuestros modelos
y_pred= regression.predict([[6.5]])

#Visualizacion de los resultados del Modelo Polinomico
#X_grid = np.arange(min(X), max(X), 0.1);
#X_grid = X_grid.reshape(len(X_grid), 1);
plt.scatter(X, Y, color = "red");
plt.plot(X, regression.predict(X), color="blue");
plt.title("Modelo de regresion SVR");
plt.xlabel('Posicion del empleado');
plt.ylabel("Sueldo en $");
plt.show();

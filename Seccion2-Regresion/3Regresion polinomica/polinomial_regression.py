#Importar librerias
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv');
X = dataset.iloc[:, 1:2].values;
Y = dataset.iloc[:,2].values;

#ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression();
lin_reg.fit(X,Y);

#Ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4);
X_poly = poly_reg.fit_transform(X);
lin_reg2 = LinearRegression();
lin_reg2.fit(X_poly, Y);

#Prediccion de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

#Visualizacion de los resultados del modelo Lienal
plt.scatter(X, Y, color = "red");
plt.plot(X, lin_reg.predict(X), color="blue");
plt.title("Modelo de regresion lineal");
plt.xlabel('Posicion del empleado');
plt.ylabel("Sueldo en $");
plt.show();

#Visualizacion de los resultados del Modelo Polinomico
X_grid = np.arange(min(X), max(X), 0.1);
X_grid = X_grid.reshape(len(X_grid), 1);
plt.scatter(X, Y, color = "red");
plt.plot(X, lin_reg2.predict(X_poly), color="blue");
plt.title("Modelo de regresion Polinomica");
plt.xlabel('Posicion del empleado');
plt.ylabel("Sueldo en $");
plt.show();

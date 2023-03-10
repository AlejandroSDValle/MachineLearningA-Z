#Importar librerias
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values;
Y = dataset.iloc[:,2].values;

#ajustar la regresion con el dataset
from sklearn.tree import DecisionTreeReegressor
regression = DecisionTreeReegressor(random_state = 0)
regression.fit(X,Y)

y_pred =regression.predict(6.5)

#Visualizacion de los resultados del Modelo Polinomico
#X_grid = np.arange(min(X), max(X), 0.1);
#X_grid = X_grid.reshape(len(X_grid), 1);
plt.scatter(X, Y, color = "red");
plt.plot(X, regression.predict(X), color="blue");
plt.title("Modelo de regresion SVR");
plt.xlabel('Posicion del empleado');
plt.ylabel("Sueldo en $");
plt.show();

#regresion lineal simple

#importar las librerias
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

dataset = pd.read_csv('Salary_Data.csv');
X = dataset.iloc[:, :-1].values;
Y = dataset.iloc[:, 1].values;

#Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.cross_validation import train_test_split;
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0);

#crear modelo de regresion lineal simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression();
regression.fit(X_train, Y_train);

#Predecir el conjunto de test
Y_pred = regression.predict(X_test);

#Visualicacion de los resultados de entrenamiento
plt.scatter(X_train, Y_train, color = "red");
plt.plot(X_train, regression.predict(X_train), color = "blue");
plt.title("Sueldo vs Anios de Experiencia (Conjunto de entrenamiento)");
plt.xlabel("Anios de Experiencia");
plt.ylabel("Sueldo en $");
plt.show();

#Visualicacion de los resultados de entrenamiento con test
plt.scatter(X_test, Y_test, color = "red");
plt.plot(X_train, regression.predict(X_train), color = "blue");
plt.title("Sueldo vs Anios de Experiencia (Conjunto de testing)");
plt.xlabel("Anios de Experiencia");
plt.ylabel("Sueldo en $");
plt.show();
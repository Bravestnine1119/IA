'''
El siguiente algoritmo realiza la identificación de un tipo de flor con base a un conjunto de datos previos.

El conjunto de datos contiene 50 muetras de cada una de tres especis de iris (iris setoda, iris virginica e iris versicolor).

El algoritmo utiliza la estructura de la neurona Perceptrón
'''
# Importamos los datos a utilizar
from sklearn.datasets import load_iris
# Importamos los módulos pandas y numpy
import pandas as pd
import numpy as np

# Importamos el módulo para representaciones gráficas
import matplotlib.pyplot as plt
# Representación gráfica de tres dimensiones del conjunto de datos
from mpl_toolkits import mplot3d
# Importamos el algoritmo de el Perceptron
from sklearn.linear_model import Perceptron

if __name__ == "__main__":
    # Cargamos el conjunto de datos
    iris_dataset = load_iris()

    # Visualizamos las etiquetas del conjunto de datos
    print(iris_dataset.target_names)

    # Leemos el conjunto de datos con la librería Pandas
    # La función np.c_ concatena las matrices a lo largo del segundo eje (se le concatena a los datos su salida)
    # también se le pasa la suma de las comlumnas de los datos más la salida
    df = pd.DataFrame(np.c_[iris_dataset['data'], iris_dataset['target']],
                      columns=iris_dataset['feature_names'] + ['target'])

    # Representación gráfica de dos dimensiones del conjunto de datos
    # Se crea una nueva figura
    # figsize se utiliza para especificar el tamaño de la figura en pulgadas
    # fig = plt.figure(figsize=(5, 5))
    #
    # plt.scatter(df["petal length (cm)"][df["target"] == 0],
    #             df["petal width (cm)"][df["target"] == 0], c="b", label="setosa")
    #
    # plt.scatter(df["petal length (cm)"][df["target"] == 1],
    #             df["petal width (cm)"][df["target"] == 1], c="r", label="versicolor")
    #
    # plt.scatter(df["petal length (cm)"][df["target"] == 2],
    #             df["petal width (cm)"][df["target"] == 2], c="g", label="virginica")
    #
    # plt.xlabel("petal_length", fontsize=14)
    # plt.ylabel("petal_width", fontsize=14)
    # plt.legend(loc="lower right", fontsize=14)
    #
    # plt.show()

    # Representación gráfica en tres dimensiones
    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection="3d")
    #
    # ax.scatter3D(df["petal length (cm)"][df["target"] == 0],
    #              df["petal width (cm)"][df["target"] == 0],
    #              df["sepal width (cm)"][df["target"] == 0], c="b")
    #
    # ax.scatter3D(df["petal length (cm)"][df["target"] == 1],
    #              df["petal width (cm)"][df["target"] == 1],
    #              df["sepal width (cm)"][df["target"] == 1], c="r")
    #
    # ax.scatter3D(df["petal length (cm)"][df["target"] == 2],
    #              df["petal width (cm)"][df["target"] == 2],
    #              df["sepal width (cm)"][df["target"] == 2], c="g", label='virginica')
    #
    # ax.set_xlabel("petal_length")
    # ax.set_ylabel("petal_width")
    # ax.set_zlabel("sepal_width")
    #
    # plt.show()

    # Reducimos el conunto de datos para entrenar el algoritmo y visualizar el resultado.
    df_reduced = df[["petal length (cm)", "petal width (cm)", "target"]]
    df_reduced = df_reduced.loc[df_reduced["target"].isin([0,1])]

    # Separamos las etiquetas de salida del resto de características del conjunto de datos
    x_df = df_reduced[["petal length (cm)", "petal width (cm)"]]
    y_df = df_reduced["target"]


    # x_df.plot.scatter("petal length (cm)", "petal width (cm)")
    # plt.show()

    # Entrenamiento del Perceptron con una única TLU.
    # max_iter especifica el número máxico de iteraciones que el modelo debe realizar en el entrenamiento.
    # random_state es una semilla aleatoria utilizada para inicializar los pesos del modelo.
    clf = Perceptron(max_iter=1000, random_state=40)
    # Aplicar el método fit sobre nuestro conjunto de datos.
    clf.fit(x_df, y_df)

    # z(x) = x1*w1 + x2w2 + b*1 función de agregación que se va a computar.
    # Se va a tratar de buscar el valor optimo de w1, w2 y b que construyen la función matemática que mejor separa los ejemplos de una clase y de otra-
    # Parametros del modelo
    print(clf.coef_)

    # Termino de interceptacion (b)
    print(clf.intercept_)

    # No resulta la siguiente ecuación: x1*0.9 + x2*1.3 + (-3)

    # Representacion grafica del limite de decision
    X = x_df.values

    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1

    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),
                         np.linspace(mins[1], maxs[1], 1000))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(10, 7))

    plt.contourf(xx, yy, Z, cmap="Set3")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k')

    plt.plot(X[:, 0][y_df == 0], X[:, 1][y_df == 0], 'bs', label="setosa")
    plt.plot(X[:, 0][y_df == 1], X[:, 1][y_df == 1], 'go', label="vesicolor")

    plt.xlabel("petal_length", fontsize=14)
    plt.ylabel("petal_width", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)

    plt.show()

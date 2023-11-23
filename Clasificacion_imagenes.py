'''

El siguiente código útiliza la base de datos del MNIST (base de datos modificada del instituto Nacional de Normas y Tecnologías).
Este código presenta la implementación de el perceptrón en el ejercicio de reconocimientos de imágenes.
Utiliza la arquitectura del perceptrón con diez TLU.

'''
# Importamos el conjutno de datos
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Para la división del conjunto de datos
from sklearn.model_selection import train_test_split
# Para el entrenamiento del conjunto de datos
from sklearn.linear_model import Perceptron

from sklearn.metrics import f1_score

if __name__ == "__main__":

    # Añadimos as_frame= False para forzar la devolucion de un array
    mnist = fetch_openml('mnist_784', as_frame=False)


    plt.figure(figsize=(20, 4))

    for index, digit in zip(range(1, 9), mnist.data[:8]):
        plt.subplot(1, 8, index)
        plt.imshow(np.reshape(digit, (28, 28)), cmap=plt.cm.gray)
        plt.title('Ejemplo: ' + str(index))
    #plt.show()

    df = pd.DataFrame(mnist.data)
    # División del conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.1)

    # Entrenamiento del algoritmo
    clf = Perceptron(max_iter=2000, random_state=40, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Número de parámetros que forman el modelo
    print(clf.coef_.shape)

    # Parámetros bias/intercept
    print(clf.intercept_)

    # Realizamos la predicción con el conjunto de datos de prueba
    y_pred = clf.predict(X_test)

    print(len(y_pred))

    # Mostramos el f1_score resultante de la clasificación
    score = f1_score(y_test, y_pred, average="weighted")
    print(f"{score*100}%")

    #Mostrando las imagenes mal clasificadas
    index = 0
    index_errors = []

    for label, predict in zip(y_test, y_pred):
        if label != predict:
            index_errors.append(index)
        index += 1

    print(len(index_errors))
    plt.figure(figsize=(20, 4))

    for i, img_index in zip(range(1, 9), index_errors[8:16]):
        plt.subplot(1, 8, i)
        plt.imshow(np.reshape(X_test[img_index], (28, 28)), cmap=plt.cm.gray)
        plt.title('Orig:' + str(y_test[img_index]) + ' Pred:' + str(y_pred[img_index]))
    plt.show()


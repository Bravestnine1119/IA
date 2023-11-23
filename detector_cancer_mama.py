'''
Este es un algoritmo enfocado a un caso práctico de la vida real.

Los datos provienen de una asociación contra el cáncer de mama de Wisconsin.

Las características de entrada se han cálculado a partir de una imagen digitalizada, de un aspirado de aguja fina
de una masa mamaria.

Se describen las características de los núcleos celulares presentes en la imagen.

'''

# Se importa la estructura de la neurona M-P.
from MPNeuron import MPNeuron
# Se importan los datos sobre las muestras de la masa mamaria.
from sklearn.datasets import load_breast_cancer
# Se importa el módulo pandas
import pandas as pd
# Se importan el módulo que permita separar los datos en datos de entrenamiento y de preubas
from sklearn.model_selection import train_test_split
# Se importa un módulo para realizar gráficas
import matplotlib.pyplot as plt
# Se importa el módulo para calcular el rango de error
from sklearn.metrics import accuracy_score
# Se importa el módulo para calcular la matriz de confusión
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    # Se reciben los datos sobre las muestras
    breast_cancer = load_breast_cancer()
    # Se guardan los datos
    X = breast_cancer.data
    # Se guardan las etiquetas de cada de dato {0,1}
    Y = breast_cancer.target
    # La función dir() muestra los atributos y métodos de un objeto
    print(dir(breast_cancer))
    # Observamos el nombre de cada uno de los tipos de datos
    #print(breast_cancer.feature_names)

    # Utilizamos una estructura de datos (dataframe) para manipular de forma más fácil los datos
    # Al dataframe primero le pasamos los datos y después el nombre de cada columna en la que se ubica cada dato
    df = pd.DataFrame(X, columns=breast_cancer.feature_names)
    #print(df)

    # Declaramos las variables que nos permitirán separa lo datos en datos de preuba y de entrenamiento
    # La función train_test_split recibe un dataframe, las etiquetas de salida
    # Con stratify manetenemos las proporciones en ambos subconjuntos con respecto a las etiquetas de saldias {0,1}
    x_train, x_test, y_train, y_test = train_test_split(df, Y, stratify=Y)
    print('--> Tamaño del conjunto de datos de entrenamiento: ', len(x_train))
    print('--> Tamaño del conjunto de datos de pruebas: ', len(x_test))


    # Ejemplo de cómo se realiza la transformación de los valores continuos a binarios
    # print(pd.cut([0.04, 2, 4, 5, 6, 0.02, 0.6], bins= 2, labels = [0,1]))
    # plt.hist([0.04, 2, 4, 5, 6, 0.02, 0.6], bins= 2)
    # plt.show()

    # Como en la neurona de M-P solo se puede trabajar con valores binarios entonces...
    # Transformamos los datos de entrenamiento y pruebas en valores binarios
    # La función .apply aplica la función pd.cut a el dataframe, separanco sus valores en dos intervalos (bins=2)
    # Los intervalos se etiquetan en dos valores [1,0] (labels=[1,0]
    x_train_bin = x_train.apply(pd.cut, bins=2, labels=[1,0])
    x_test_bin = x_test.apply(pd.cut, bins=2, labels=[1,0])

    # Instanciamos el model MPNeuron
    mp_neuron = MPNeuron()
    # Encontramos el threshold óptimo
    mp_neuron.fit(x_train_bin.to_numpy(), y_train)
    # Realizamos las predicciones para ejemplos nuevo sque no se encuentran en el conjunto de datos de entrenamiento
    y_pred = mp_neuron.predict(x_test_bin.to_numpy())
    #print(y_pred)
    # Calculamos la exactitud de nuestra predicción
    print(f'Exactitud de la predicción: {accuracy_score(y_test, y_pred)*100}%')
    # Calculamos la matriz de confusión
    term = confusion_matrix(y_test, y_pred)
    print(f"--> El modelo predice correctamente que una muestra pertenece a la clase negativa (TN): {term[0][0]}.\n")
    print(f"--> El modelo predice incorrectamente que una muestra pertenece a la clase positiva cuando en realidad pertenece a la clase negativa (FP): {term[0][1]}.\n")
    print(f"--> El modelo predice incorrectamente que una muestra pertenece a la clase negativa cuando en realidad pertenece a la clase positiva (FN): {term[1][0]}.\n")
    print(f"--> El modelo predice correctamente que una muestra pertenece a la clase positiva (TP): {term[1][1]}.\n")







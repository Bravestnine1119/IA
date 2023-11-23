'''
Este programa toma como referencia a la primera neruona artificial de la historia, la neurona de McCulloch y Pitts.

Se caracteriza por recibir uno o más valores binarios {1,0} y retorna otro valor binario {1,0}.

Activa su salida cuando más de un número de valores de entrada se encuentran activos.

Debe establecerse manualmente el número de valores que deben estar activos, a este valor se le denomina "threshold".

LIMITACIONES:
- Recibe únicamente valores binarios {1,0}.
- Requiere la selecicón del threshols de manera manual.
- Todas las entradas son iguales. No se le puede asignar un mayor peso a una de las entradas.
- No son capaces de resolver problemas que no sean linealmente separables.

'''
import numpy as np
from sklearn.metrics import accuracy_score

# Clase en función de la arquitectura de la neurona M-P
class MPNeuron:
    # Constructor en donde se declara el threshold
    def __init__(self):
        self.threshold = None

    # A través de un array se realiza la suma de sus valores
    def model(self, x):
        # Se regresa True si el total de la suma es mayor al threshold
        # Si el total de la suma es menor al threshold se regresa False
        return (sum(x) >= self.threshold)

    # Funció que recibe un número listas, las evalúa y regresa un array con el resultado de su evaluación
    def predict(self, X):
        # Se declara un lista de apoyo.
        Y = []
        # Se recorre cada una de las listas envíadas.
        for x in X:
            # Se evalúa la lista
            result = self.model(x)
            # Se agrega el resultado a la lista de apoyo
            Y.append(result)
        # La lista se convierte en un array de numpy y se regresa
        return np.array(Y)

    # Función para asignar un threshold de manera automática
    def fit(self, X, Y):
        # La función recibe los datos y las características de salida
        accuracy = {}
        # Seleccionamos un threshold entre el # de características de entrada
        # La función shape[] se utiliza para obtener el número de elementos en cada dimensión ([1]]
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        # Seleccionamos el threshold que mejores resultado proporciona
        self.threshold = max(accuracy, key=accuracy.get)
        print(accuracy)
        print(f"Threshold elegido: {self.threshold}")

if __name__ == '__main__':
    # Instanciamos la neurona
    mp_neuron = MPNeuron()

    # Establecemos un threshold
    mp_neuron.threshold = 3

    # Evalúamos diferentes casos de uso
    evaluacion = mp_neuron.predict([[1,0,0,0,0], [1,1,1,1], [1,1,1,0]])

    # Imprimimos el resultado de la evaluacón
    print(evaluacion)

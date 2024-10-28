import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image

objetivo = Image.open("demo.jpg").resize((20, 20))  
objetivo = np.array(objetivo)  

TAMANO_POBLACION = 200
GENERACIONES = 100
TASA_MUTACION = 0.01


def crear_individuo():
    return np.random.randint(0, 256, (20,20), dtype=np.uint8)

def fitness(individuo):
    return -np.sum(np.abs(individuo - objetivo)) 


def seleccion(poblacion):
    poblacion = sorted(poblacion, key=lambda x: fitness(x), reverse=True)
    return poblacion[:2]  

def cruce(padre1, padre2):
    hijo = np.copy(padre1)
    mask = np.random.randint(0, 2, (20,20)).astype(bool)  
    hijo[mask] = padre2[mask]
    return hijo

# Mutación: Alterar algunos píxeles aleatoriamente
def mutacion(individuo):
    mutante = np.copy(individuo)
    for i in range(mutante.shape[0]):
        for j in range(mutante.shape[1]):
            if random.random() < TASA_MUTACION:
                mutante[i, j] = np.random.randint(0, 256)
    return mutante

# Algoritmo Genético Principal
def algoritmo_genetico():
    # Crear población inicial
    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]

    for generacion in range(GENERACIONES):
        padres = seleccion(poblacion)  # Selección
        nueva_poblacion = []

        # Generar nuevos individuos mediante cruce y mutación
        for _ in range(TAMANO_POBLACION):
            hijo = cruce(*padres)
            hijo = mutacion(hijo)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion  # Actualizar la población

        # Mostrar el mejor individuo cada 50 generaciones
        if generacion % 50 == 0:
            mejor = seleccion(poblacion)[0]
            print(f"Generación {generacion}, Fitness: {fitness(mejor)}")
            plt.imshow(mejor)
            plt.show()

    # Devolver el mejor individuo al final
    mejor = seleccion(poblacion)[0]
    return mejor

# Ejecutar el AG
resultado = algoritmo_genetico()

# Mostrar la imagen final generada
plt.imshow(resultado)
plt.title("Imagen Generada")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

objetivo = Image.open("sebas.jpg").convert("L").resize((128, 128)) 
objetivo = np.array(objetivo)

TAMANO_POBLACION = 500 
GENERACIONES = 5000
TASA_MUTACION = 0.01

alpha = 0.7
beta = 0.3   

def crear_individuo():
    return np.random.randint(0, 256, (128, 128), dtype=np.uint8)

def fitness(individuo):
    valor_psnr = psnr(objetivo, individuo) 
    valor_ssim = ssim(objetivo, individuo)  
    return alpha * valor_psnr + beta * valor_ssim  


def seleccion(poblacion):
    poblacion = sorted(poblacion, key=lambda x: fitness(x), reverse=True)
    return poblacion[:2] 

def cruce(padre1, padre2):
    hijo = np.copy(padre1)
    mask = np.random.randint(0, 2, (128, 128)).astype(bool) 
    hijo[mask] = padre2[mask]
    return hijo

def mutacion(individuo):
    mutante = np.copy(individuo)
    for i in range(mutante.shape[0]):
        for j in range(mutante.shape[1]):
            if random.random() < TASA_MUTACION:
                mutante[i, j] = np.random.randint(0, 256) 
    return mutante

def algoritmo_genetico():
    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]

    for generacion in range(GENERACIONES):
        padres = seleccion(poblacion) 
        nueva_poblacion = []

        for _ in range(TAMANO_POBLACION):
            hijo = cruce(*padres)
            hijo = mutacion(hijo)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion 

        if generacion % 50 == 0:
            mejor = seleccion(poblacion)[0]
            print(f"Generación {generacion}, Fitness: {fitness(mejor)}")
            plt.imshow(mejor, cmap="gray")
            plt.show()

    mejor = seleccion(poblacion)[0]
    return mejor

resultado = algoritmo_genetico()

plt.imshow(resultado, cmap="gray")
plt.title("Imagen Generada Mejorada")
plt.show()

import random 

#representamos a bolivia como un espacio de dimensiones
#donde cada incide representa a un departamento y su valor representa una asginacion a ese departamento
# [0,1,2,3,4,5,6,7,8,9]
# La paz, Oruro, Potosi, Cochabamba, Tarija, Chuquisaca, Pando, Beni, Santa Cruz
# El dominio estará representado por colores R,V,A (Rojo verde y azul)
# ejemplo de asignación de valores de dominio a variables
# [R,V,A,V,R,V,V,A,A]

#definimos nuestras adyacencias

bolivia = {
    'Pando': ['La Paz', 'Beni'],
    'Beni': ['Pando', 'Santa Cruz', 'Cochabamba', 'La Paz'],
    'Santa Cruz': ['Chuquisaca', 'Cochabamba', 'Beni'],
    'Tarija': ['Chuquisaca', 'Potosi'],
    'Chuquisaca': ['Tarija', 'Cochabamba', 'Potosi', 'Santa Cruz'],
    'Cochabamba': ['La Paz', 'Oruro', 'Potosi', 'Chuquisaca', 'Santa Cruz', 'Beni'], 
    'La Paz': ['Pando' , 'Beni', 'Oruro', 'Santa Cruz'],
    'Oruro' : ['Potosi', 'Cochabamba', 'La Paz', ],
    'Potosi': ['Oruro', 'Cochabamba', 'Potosi', 'Tarija']
}

departamentos = ['Pando', 'Beni', 'La Paz', 'Tarija', 'Chuquisaca', 'Cochabamba', 'La Paz', 'Oruro', 'Potosi']

indices = {'Pando': 0, 'Beni': 1, 'Santa Cruz': 2, 'Tarija': 3, 'Chuquisaca': 4
           , 'Cochabamba': 5, 'La Paz': 6, 'Oruro': 7, 'Potosi': 8,}

dominio = ['R', 'V', 'A']

def get_aptitud(individuo):
    conflicto = 0
    #verificamos los conflictos entre departamentos
    for i in range(len(individuo)):
        departamento = departamentos[i]
        color_departamento = individuo[i]
        for vecino in bolivia[departamento]:
            color_vecino = individuo[indices[vecino]]
            if color_departamento == color_vecino:
                conflicto += 1
    return conflicto

def crossover(padre, madre):
    punto = random.randint(1, len(padre) - 1)
    hijo = padre[:punto] + madre[punto:]
    return hijo

def seleccion(poblacion, res_aptitud, tam_torneo=2):
    torneo = random.sample(range(len(poblacion)), tam_torneo)
    mejor_indice = torneo[0]
    for i in torneo[1:]:
        if res_aptitud[i] < res_aptitud[mejor_indice]:
            mejor_indice = i
    return poblacion[mejor_indice]

#ajustamos la funcion de mutacion
def mutacion(individuo, tasa_mutacion):
    for indiv in range(len(individuo)):
        if random.random() < tasa_mutacion:
            #escogemos sobre el dominio actual
            individuo[indiv] = random.choice(dominio)
    return individuo


def programa(n, tam_poblacion, generaciones, tasa_mutacion):
    poblacion = []
    for x in range(tam_poblacion):
        # generamos poblacion inicial en funcion de los valores de dominio
        # y del tamaño de nuestro espacio de variables
        cromosoma = [random.choice(dominio) for y in range(len(departamentos))]
        poblacion.append(cromosoma)
    ## lo de arriba solo genera poblacion inicial
    ## ahora generamos nueva poblacion
    for generacion in range(generaciones): #iteramos sobre las generaciones
        #capturamos los valores de la funciuon fitness 
        res_aptitud = [get_aptitud(individuo) for individuo in poblacion]
        ######## la funcion aptitud todavia no la tenemos y la sacamos de un cromosoma
        ######## pero la idea es que saque la aptitud de todos los individuos de la poblacion
        ######## ordenamos y agarramos los cromosomas con mayor valor de aptitud
        padres = []
        for y in range(tam_poblacion // 2):
            #hacemos la mitad de la poblacion por que capturamos padre y madre
            padre = seleccion(poblacion, res_aptitud)
            ## selecciono el padre utilizando la poblacion y el resultado de la funcion de aptitud
            madre = seleccion(poblacion, res_aptitud)
            padres.append(padre)
            padres.append(madre)
        
        ##hacemos operaciones para cruzar parejas
        result = []
        for i in range(0, tam_poblacion, 2):
            hijo1 = crossover(padres[i], padres[i+1])
            hijo2 = crossover(padres[i+1], padres[i])
            result.append(mutacion(hijo1, tasa_mutacion))
            result.append(mutacion(hijo2, tasa_mutacion))
        
        poblacion = result
        mejor = min(poblacion, key=get_aptitud)
        if get_aptitud(mejor) == 0:
            return mejor
    return mejor

if __name__ == "__main__":
    n = 9
    tam_poblacion = 100
    generaciones = 1000
    tasa_mutacion = 0.01
    solucion = programa(n, tam_poblacion, generaciones, tasa_mutacion)
    print(solucion)
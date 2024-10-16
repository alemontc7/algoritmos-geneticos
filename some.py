import random 

def get_aptitud(individuo):
    conflicto = 0
    for i in range(len(individuo)):
        for j in range(i +  1, len(individuo)):
            if abs( i - j ) == abs(individuo[i] - individuo[j]):
                conflicto+=1
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

def mutacion(individuo, tasa_mutacion):
    for indiv in range(len(individuo)):
        if random.random() < tasa_mutacion:
            individuo[indiv] = random.randint(0, len(individuo) - 1)
    return individuo


def programa(n, tam_poblacion, generaciones, tasa_mutacion):
    poblacion = []
    for x in range(tam_poblacion):
        cromosoma = [random.randint(0, n-1) for y in range(n)]
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
    n = 8
    tam_poblacion = 100
    generaciones = 1000
    tasa_mutacion = 0.01
    solucion = programa(n, tam_poblacion, generaciones, tasa_mutacion)
    print(solucion)
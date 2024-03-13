# ------------- fara matrice de adiacenta ---------------
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# constante
x = [0,3,6,7,15,10,16,5,8,1.5]                      # X coord
y = [1,2,1,4.5,-1,2.5,11,6,9,12]                    # y coord
cities  = ["Gliwice", "Cairo", "Rome", "Krakow", "Paris", "Alexandria", "Berlin", "Tokyo", "Hong Kong", "Rio"]
N = len(cities)                                     # number of cities
P = 250                                             # indivizi per generatie
G = 150                                             # number of generations used
solution = None                                     # store the result //// np.array(N, dtype=int) 
population = None                                   # store the population
keep = 40                                           # cati indivizi vor fi pastrati pentru noua generatie
C = int(P * 0.25)                                   # crossover
M = int(P * N * 0.03)                               # mutatii
flag = True

# calcularea distantelor dintre puncte pentru o solutie
def fitness(individ):
    dist = 0
    for i in range(N - 1):
        curr  = individ[i]
        next  = individ[i+1]
        dist += math.sqrt((x[curr] - x[next])**2 + (y[curr] - y[next])**2)

    dist += math.sqrt((x[0] - x[N-1])**2 + (y[0] - y[N-1])**2)

    return dist   


# initializare populatie
def initialize():
    global population
    population = np.zeros((P, N), dtype=int)        # initialize population 
    for i in range(P):
        aux = np.arange(stop=N, dtype=int)
        np.random.shuffle(aux)
        population[i,:] = aux.copy()


# verifica daca exista aceleasi elemente intre doi vectori
def check_similarity(array1, array2):
    union = np.union1d(array1, array2)
    return (len(union) == len(array1))


# crossover dintre 2 indivizi
def crossover(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()

    max_tries = 5                   # pentru a nu testa la infinit valori random
    ok = True
    
    while ok and max_tries:
        r = random.randint(0, N-1)
        max_tries -= 1
        aux1 = child1[r : N-1].copy()
        aux2 = child2[r : N-1].copy()
        if check_similarity(aux1, aux2): 
            ok = False
            child1[r : N-1] = aux2
            child2[r : N-1] = aux1

    return child1, child2


# mutatie individ
def mutation(parent):
    child = parent.copy()
    index1 = random.randint(0, N-1)
    index2 = index1
    
    while index2 == index1:
        index2 = random.randint(0, N-1)

    child[index1], child[index2] = child[index2], child[index1]
    return child


# show progress
def draw():
    plt.clf()

    # Trasează punctele
    plt.scatter(x, y)

    # Adaugă denumirile punctelor
    for i, city in enumerate(cities):
        plt.annotate(city, (x[i], y[i]))

    for i in range(N-1):
        global solution
        curr = solution[i]
        next = solution[i+1]
        plt.plot([x[curr], x[next]], [y[curr], y[next]], 'k-')

    plt.plot([x[solution[0]], x[solution[N-1]]], [y[solution[0]], y[solution[N-1]]], 'k-')

    # Afișează graficul
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ruta optima')
    plt.grid(True)
    plt.show()


# every generation
def new_generation():
    distances = np.zeros(P)         # distanta + probabilitatea asociata ei pe disc
    global population
    new_population = population.copy()

    for i in range(P):
        distances[i] = fitness(population[i,:])
    
    # sort in fct de distances
    ok = False
    while not ok:
        ok = True
        for i in range(P-1):
            if distances[i] > distances[i+1]:           # swap indivizi si distantele lor
                distances[i],    distances[i+1]    =  distances[i+1],   distances[i]
                population[i,:], population[i+1,:] = population[i+1,:], population[i,:]
                ok = False

    minim = distances[0]
    maxim = distances[P-1]

    global solution
    global flag
    if flag == True or fitness(solution) > fitness(population[0,:]): 
        solution = population[0,:].copy()
        flag = False

    norm_distances = distances.copy()
    for i in range(P):
        norm_distances[i] = (norm_distances[i] - minim) / (maxim - minim)

    index = 0

    # pastrati pentru urmatoare generatie
    for k in range(keep):         # k --- count
        r1 = random.random()
        suma1 = 0

        for i in range(P):
            if suma1 <= r1 and suma1 + norm_distances[i] >= r1:
                new_population[index,:] = population[i,:].copy()
                index += 1
                break

            suma1 += norm_distances[i]
    #print("done copy")

    # crossover
    for c in range(C):          # c --- count
        r1 = random.random()
        child1 = None
        child2 = None
        suma1 = 0
        ok = False
        for i in range(P):
            if suma1 <= r1 and suma1 + norm_distances[i] >= r1:
                r2 = random.random()
                suma2 = 0
                j = 0
                while j < P:
                    if suma2 <= r2 and suma2 + norm_distances[j] >= r2:
                        if i == j:
                            r2 = random.random()
                            j = 0
                        else:
                            child1, child2 = crossover(population[i,:], population[j,:])
                            new_population[index,:]   = child1.copy()
                            new_population[index+1,:] = child2.copy()
                            index += 2
                            ok = True
                            break
                            
                    suma2 += norm_distances[j]
                    j += 1

            suma1 += norm_distances[i]
            if ok:
                break
    #print("done crossover")            

    # mutatie
    for m in range(M):              # m --- count
        r1 = random.random()
        suma1 = 0

        for i in range(P):
            if suma1 <= r1 and suma1 + norm_distances[i] >= r1:
                child = mutation(population[i,:])
                new_population[index,:] = child.copy()
                index += 1
                break

            suma1 += norm_distances[i]
    #print("done mutation")

    while index < P:
        aux = np.arange(stop=N, dtype=int)
        np.random.shuffle(aux)
        new_population[index,:] = aux.copy()
        index += 1
    
    #print("done copy")

    population = new_population.copy()


# call functions
initialize()

for g in range(G):
    print(g, ".\n")
    if g % 10 == 1:
        draw()
    new_generation()
    print("dist:", fitness(solution))

draw()

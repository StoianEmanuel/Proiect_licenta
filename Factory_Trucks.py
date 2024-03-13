import numpy as np
import random
import math
import matplotlib.pyplot as plt
from utils import matrice_adiacenta


numVehicles    = 4       # number of vehicles
numCustomers   = 10      # number of customers
maxTrCapacity  = 10
numGeneration  = 100     # number of generations
populationSize = 75      # population size
parentsKept    = 10
numCrossovers  = int(populationSize * 0.3)
numMutations   = int(populationSize * numCustomers * 0.015)
best_routes = np.zeros((numVehicles,numCustomers), dtype=int)
flag = True

x = np.array([0,3,6,7,15,10,16,5,8,1.5])        # customers locations for printing on screen
y = np.array([1,2,1,4.5,-1,2.5,11,6,9,12])
depot_x = 4             # location of depot
depot_y = 3
costCustomers = np.array([1,5,6,2,3,7,1,1,2,2])       # cost per visiting customer
matrAdiacenta = matrice_adiacenta(depot_x, depot_y, x, y)
population  = None


# va trebui mecanism pentru a ma asigura ca sunt folositi toti clientii

# does what it says
def generate_individ(trucks, capacity, customers, cost):
    individ = np.zeros((trucks, customers))
    costPerTruck = np.zeros(trucks)

    retry = True
    while retry == True:
        remaining = 0
        
        for i in range(customers):
            ok = False
            for j in range(int(customers/2)):
                r = random.randint(0, trucks-1)
                if costPerTruck[r] + cost[i] <= capacity:
                    individ[r][i] = 1
                    costPerTruck[r] += cost[i]
                    ok = True
                    break
                else:
                    ok = False
            
            if not ok:
                for j in range(customers):
                    if costPerTruck[j] + cost[i] <= capacity:
                        individ[r][i] = 1
                        costPerTruck[r] += cost[i]
                        ok = True
                        break
            
                if not ok:
                        remaining += 1

        if remaining == 0:
            retry = False

    return individ


# show progress
def draw():
    plt.clf()

    global best_routes
    dimY, dimX = best_routes.shape

    # Trasează punctele
    global costCustomers
    plt.scatter(x, y, color='black')

    # Adaugă costurile ca și string la fiecare punct
    for i, (x_point, y_point) in enumerate(zip(x, y)):
        plt.text(x_point, y_point, str(costCustomers[i]), fontsize=10, ha='center', va='bottom')

    # Trasează punctul depot
    global depot_x
    global depot_y
    plt.scatter(depot_x, depot_y, color='orange', label='Depot')
    plt.text(depot_x, depot_y, "Depot", fontsize=10, ha='center', va='bottom')

    colors = ["red", "blue", "green", "yellow"]
    for i in range(dimY):
        first = -1
        last = -1
        for j in range(dimX):
            if best_routes[i][j] == 1:
                if first == -1:
                    first = j
                
                if last != -1:
                    plt.plot([x[j], x[last]], [y[j], y[last]], color = colors[i])

                last = j

        plt.plot([depot_x, x[first]], [depot_y, y[first]], color = colors[i])
        plt.plot([depot_x, x[last]],  [depot_y, y[last]],  color = colors[i])

    # Afișează graficul
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rute camioane')
    plt.grid(True)
    plt.show()


# only for initialization
def initialize_population(trucks, customers, populationSize, capacity, cost):
    population = np.zeros((populationSize, trucks, customers))
    for i in range(populationSize):
        population[i] = generate_individ(trucks, capacity, customers, cost)

    return population


# sum of distances for each truck
def fitness(individ, cost, matr_adiacenta, max_capacity):
    totalDist = 0
    dimY, dimX = individ.shape
    capacity_valid = True

    for i in range(dimY):       # from + to depot
        capacity = 0
        first = -1
        last = -1
        for j in range(dimX):
            if individ[i][j] == 1:
                if last != -1:
                    totalDist += matr_adiacenta[last][j]
                    
                if first == -1:
                    first = j

                last = j
                capacity += cost[j]
        
        totalDist = totalDist + matr_adiacenta[0][first] + matr_adiacenta[0][last]
        
        if capacity > max_capacity:
            capacity_valid = False
    
    if not capacity_valid:
        totalDist *= 100
    
    return totalDist


# return 2 children as the result of crossover operator
def crossover(parent1, parent2):
    crossoverPoint = random.randint(0, len(parent1[0])-1)
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent1)

    child1[:, :crossoverPoint] = parent1[:, :crossoverPoint]
    child1[:, crossoverPoint:] = parent2[:, crossoverPoint:]
    
    child2[:, :crossoverPoint] = parent2[:, :crossoverPoint]
    child2[:, crossoverPoint:] = parent1[:, crossoverPoint:]
    
    return child1, child2


# return mutated child
def mutation(parent):
    mutationPoint = random.randint(0, len(parent[0])-1)
    child = parent.copy()
    np.random.shuffle(child[:, mutationPoint])
    return child


# create new generation
def new_generation(populationSize, numVehicles, capacity, numCustomers, cost, parentsKept, numCrossovers, numMutations, matr_adiacenta):
    distances = np.zeros(populationSize)         # distanta + probabilitatea asociata ei pe disc
    
    global population
    new_population = population.copy()

    for i in range(populationSize):
        distances[i] = fitness(population[i,:], cost, matr_adiacenta, capacity)
    
    # sort in fct de distances
    ok = False
    while not ok:
        ok = True
        for i in range(populationSize-1):
            if distances[i] > distances[i+1]:           # swap indivizi si distantele lor
                distances[i],    distances[i+1]    =  distances[i+1],   distances[i]
                population[i,:], population[i+1,:] = population[i+1,:], population[i,:]
                ok = False

    minim = distances[0]
    maxim = distances[populationSize-1]

    global best_routes
    global flag
    if flag == True or fitness(best_routes, cost, matr_adiacenta, capacity) > fitness(population[0,:], cost, matr_adiacenta, capacity): 
        best_routes = population[0,:].copy()
        flag = False

    norm_distances = distances.copy()
    for i in range(populationSize):
        norm_distances[i] = (norm_distances[i] - minim) / (maxim - minim)


    index = 0

    # pastrati pentru urmatoare generatie
    for k in range(parentsKept):         # k --- count
        r1 = random.random()
        suma1 = 0

        for i in range(populationSize):
            if suma1 <= r1 and suma1 + norm_distances[i] >= r1:
                new_population[index,:] = population[i,:].copy()
                index += 1
                break

            suma1 += norm_distances[i]
    #print("done copy")

    # crossover
    for c in range(numCrossovers):          # c --- count
        r1 = random.random()
        child1 = None
        child2 = None
        suma1 = 0
        ok = False
        for i in range(populationSize):
            if suma1 <= r1 and suma1 + norm_distances[i] >= r1:
                r2 = random.random()
                suma2 = 0
                j = 0
                while j < populationSize:
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
    for m in range(numMutations):              # m --- count
        r1 = random.random()
        suma1 = 0

        for i in range(populationSize):
            if suma1 <= r1 and suma1 + norm_distances[i] >= r1:
                child = mutation(population[i,:])
                new_population[index,:] = child.copy()
                index += 1
                break

            suma1 += norm_distances[i]
    #print("done mutation")

    while index < populationSize:
        aux = generate_individ(numVehicles, capacity, numCustomers, cost)
        new_population[index,:] = aux.copy()
        index += 1
    
    #print("done copy")

    population = new_population.copy()


# program itself
population = initialize_population(trucks = numVehicles, customers = numCustomers, populationSize = populationSize, capacity = maxTrCapacity, cost = costCustomers)

for g in range(numGeneration):
    print(g, ".\n")
    if g % 10 == 1:
        draw()
    new_generation(populationSize = populationSize, numVehicles = numVehicles, capacity = maxTrCapacity, numCustomers = numCustomers, cost = costCustomers,
                   parentsKept = parentsKept, numCrossovers = numCrossovers, numMutations = numMutations, matr_adiacenta = matrAdiacenta)
    print("dist:", fitness(best_routes, costCustomers, matrAdiacenta, maxTrCapacity))

print(best_routes)
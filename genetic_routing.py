#! C:\Users\manue\Desktop\GA\Proiect\venv\Scripts\python.exe
from a_star_routing import multiple_routes_A_star, read_file_routes, mark_areas_in_grid
from utils import fitness_function, delete_file, simplify_path_list
import numpy as np
import random
import copy

# Genetic Algorithms param
POPULATION_SIZE:    int
ROUNDS:             int
MUTATIONS:          int
CROSSOVERS_1P:      int
PARENTS_KEPT:       int

# File specific param
AXIS_MULT = None
ROWS:               int
COLUMNS:            int
OFFSET_X = 0
OFFSET_Y = 0

# Variables used for finding the best solution
population = []
template_grid = None
routes = [] # global to ensure that every function has access to it
nr_needed_routes = 0 # how to ignore the routes that are defined but not used


# maybe i will insert in mutation / crossover a way to move a cell to adjent one in order to reduce some angles
class Individual:
        def __init__(self, order = [], grid = [], paths = [], unplaced_routes_number = 0, paths_total_cost = -1):
            self.order  = order    # holds order of routes from input
            self.grid   = grid    # grid might be optional
            self.paths  = paths    # [[path1], [path2], ...]
            self.unplaced_routes_number = unplaced_routes_number # from the necessary ones
            self.total_cost = paths_total_cost      # fitness_value; if no paths would be 0

        def __repr__(self):
            return repr((self.order, self.grid, self.paths, self.unplaced_routes_number, self.total_cost))
        
        @staticmethod
        def return_path_cost(self):
            return self.total_cost

        def __str__(self) -> str:
            return f'Order: {self.order}; Total cost: {self.total_cost}'

best_solution = Individual()


# assign a random order for routes to be placed from 0 to route list_size - 1
def random_order_list(list_size: int):
    order_list = list(range(list_size))
    random.shuffle(order_list)
    return order_list


# return a list sorted according to another one containing indexes
def order_list(original_list, indexes_list:list):
    modified_list = copy.deepcopy(original_list)
    for i in range(len(indexes_list)):
        index = indexes_list[i]
        modified_list[i], modified_list[index] = modified_list[index], modified_list[i]
    return modified_list


# return data needed for Individual type object (order of routing - list, routes placed - array, grid with areas used / blocked)
def generate_individual(ordered_index_routes = None, routes_number = 0): # maybe add option for a starting grid and paths
    global routes, nr_needed_routes
    if ordered_index_routes:
        random_order_indexes = ordered_index_routes
    else:
        random_order_indexes = random_order_list(nr_needed_routes)

    ordered_routes = order_list(original_list=routes, indexes_list = random_order_indexes)
    # routes number will be used to either remove cycles or will be dropped
    global template_grid, ROWS, COLUMNS
    grid, possible_solution = multiple_routes_A_star(grid = template_grid, routes = ordered_routes, pins_sizes = 1, 
                                                         rows = ROWS, columns = COLUMNS)
    return Individual(order=random_order_indexes, 
                      paths=possible_solution, grid=grid)  


# initialize population; used for first generation
def initialize_population(population_size: int, routes_number):
    global template_grid, routes
    population = []

    for i in range(population_size):
        value = generate_individual(routes_number=routes_number)
        population.append(value)
    
    return population


# return a value in range 0 - (n-1) that indicates how many elements are in common of the same index; 
# return value: 0 - nothing in common, n-1 - everything
def common_part_index(list1, list2):
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return i
    return len(list1)


# operations used for crossover; it is applied directly onto the children
def crossover_1P_operator(child1: Individual, child2: Individual):
    size = len(child1.order)
    index = random.randint(0, size - 1)

    child1_order = child1.order
    child2_order = child2.order
    child1.order = child1_order[:index] + child2_order[index:]
    child2.order = child2_order[:index] + child1_order[index:]
    
    if len(child1.order) != len(set(child1.order)):
        return False

    child1 = generate_individual(ordered_index_routes = child1.order)
    child2 = generate_individual(ordered_index_routes = child2.order)

    return True


# max 5 tries to find 2 suitable parents for a 1 point crossover else returns the parents values; returns 2 children
def crossover():
    global population, POPULATION_SIZE
    tries = 5
    
    index1 = random.randint(0, POPULATION_SIZE-1)
    index2 = random.randint(0, POPULATION_SIZE-1)
    
    while tries > 0:
        tries -= 1

        index1 = random.randint(0, POPULATION_SIZE-1)
        index2 = random.randint(0, POPULATION_SIZE-1)
        while index1 == index2:
            index2 = random.randint(0, POPULATION_SIZE-1)
        
        parent1 = population[index1]
        parent2 = population[index2]

        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
            
        crossover_success = crossover_1P_operator(child1, child2)
        if crossover_success == True:
            return child1, child2

    return copy.deepcopy(population[index1]), copy.deepcopy(population[index2])


# perform a mutation on a individual and return the child
def mutation(parent: Individual):       # later can be modified to replace one segment from a circuit to another (3 or more points routing)
    order = copy.deepcopy(parent.order)
    size = len(order)
    index1 = random.randint(0, size-1)
    index2 = random.randint(0, size-1)

    while index1 == index2:
        index2 = random.randint(0, size-1)
    
    order[index1], order[index2] = order[index1], order[index2] # swap the order for routing

    # to add a starting matrix from 0 to index
    child = generate_individual(ordered_index_routes=order)
    
    return child



# Selection based on roulette wheel; returns list on indexes of individuals selected from a generation
def roulette_wheel_selection(population, selected_number):
    # Sum of all fitness values of inividuals
    sum_fitness = sum(individual.total_cost for individual in population)

    selected = []
    for _ in range(selected_number):
        # Choose a point on roulette wheel
        selection_point = random.uniform(0, sum_fitness)
        sum_selection = 0
        index = 0
        for individual in population:
            sum_selection += individual.total_cost
            if sum_selection >= selection_point:
                selected.append(index)
                break

    return selected


# Adjusts the parameters (population size, number of generations, crossover and mutation operations per generatiosn, parents kept
# read the file -- find size of the matrix used for representation, offset on X, Y, axis multiplying factor, areas to be avoided
def set_parameters_GA(file_name: str, population_coef = 2, rounds_coef = 3, 
                      parents_coef = 35, crossover_1P_coef = 10, mutations_coef = 10):
    global routes, nr_needed_routes
    routes, colors = read_file_routes(file_name=file_name, draw = False)
    nr_needed_routes = len(routes)
    blocked_areas = None

    # from file find x y coordinates for grid, if there are no such values, extract from min max
    global ROWS, COLUMNS, OFFSET_X, OFFSET_Y, AXIS_MULT
    OFFSET_X = 0
    OFFSET_Y = 0
    AXIS_MULT = 1

    global POPULATION_SIZE, ROUNDS, PARENTS_KEPT, CROSSOVERS_1P, MUTATIONS
    POPULATION_SIZE = min(int(nr_needed_routes * population_coef), 50) # 55
    ROUNDS = min(int(nr_needed_routes * rounds_coef), 50) # 25
    ROWS = 55
    COLUMNS = 55

    PARENTS_KEPT = int((POPULATION_SIZE * parents_coef) / 100) # 25
    CROSSOVERS_1P = int((POPULATION_SIZE * crossover_1P_coef) / 100) # 5
    MUTATIONS = int((nr_needed_routes * mutations_coef) / 100) # 10

    global template_grid
    template_grid = np.zeros((ROWS, COLUMNS), dtype=int)
    template_grid = mark_areas_in_grid(blocked_cells = blocked_areas, grid = template_grid, value = 0)

    #individual_routes_placed = [[[] for j in range(routes_number)] for i in range(POPULATION_SIZE)] - instead of individual class
    #individual_grid = [[[] for j in range(routes_number)] for i in range(POPULATION_SIZE)]



# function that performs the routing with GA based on the parameters from first_generation()
def genetic_routing(file_name, event = None):
    file1 = open(file_name, "a")

    global population, best_solution, POPULATION_SIZE, ROUNDS, PARENTS_KEPT
    for i in range(ROUNDS):
        if not event or not event.is_set():
            print(f"Generation {i+1}")
            for j in range(POPULATION_SIZE):
                if population[j].total_cost == -1:
                    paths = population[j].paths
                    n = population[j].unplaced_routes_number
                    cost = fitness_function(routes = paths, 
                                            unplaced_routes_number = n, 
                                            unplaced_route_penalty=1.5)
                    population[j].total_cost = cost
                
            population.sort(reverse = False, key = Individual.return_path_cost)
        
            if best_solution.total_cost == -1 or \
                (population[0].total_cost != -1 and population[0].total_cost < best_solution.total_cost):
                best_solution = copy.deepcopy(population[0])

            file1.write(f"{i}: {str(best_solution)}\n")
            best_paths = simplify_path_list(best_solution.paths)
            if best_paths:
                file1.write(f"{best_paths}\n")

            next_generation = []

            indexes = roulette_wheel_selection(population=population, selected_number=PARENTS_KEPT)
            for index in indexes:
                next_generation.append(copy.deepcopy(population[index]))

            for j in range(0, CROSSOVERS_1P):
                child1, child2 = crossover()
                next_generation.append(child1)
                next_generation.append(child2)
            
            for j in range(0, MUTATIONS):
                index = random.randint(0, POPULATION_SIZE-1)
                child = mutation(population[index])
                next_generation.append(child)
            
            # maybe add another type of mutation or crossover operators

            # add new individuals into the new generation
            count_new_individuals = POPULATION_SIZE - PARENTS_KEPT - CROSSOVERS_1P * 2 - MUTATIONS
            try:
                for j in range(count_new_individuals):
                    individual = generate_individual()
                    next_generation.append(individual)
            except Exception as e:
                print("Error while adding new individuals; Error:", e) 
            
            #for j in range(POPULATION_SIZE):   file1.write(f"{i}, {j}, {next_generation[j].order}, {next_generation[j].total_cost}\n")
            population = next_generation

    file1.close()



def run_genetic_algorithm(save_file, read_file, event = None):
    if not event or not event.is_set():
        delete_file(file_name = save_file)
        # population_percentage = 0
        # generations = 0
        # crossover_1P_percentage = 0
        # mutation_percentage = 0

        set_parameters_GA(file_name = read_file)

        print("Initialize population")
        global population, POPULATION_SIZE, nr_needed_routes
        population = initialize_population(population_size = POPULATION_SIZE, routes_number = nr_needed_routes)
        
        genetic_routing(file_name = save_file, event = event)
        print(f"\n\norder = {best_solution.order}, cost = {best_solution.total_cost}, paths = {best_solution.paths}")
        
        if event:
            event.set()


if __name__ == "__main__":
    run_genetic_algorithm(save_file = "solution.txt", read_file = "pins.csv")
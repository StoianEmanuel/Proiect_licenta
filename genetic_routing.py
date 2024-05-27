#! C:\Users\manue\Desktop\GA\Proiect\venv\Scripts\python.exe
from routing_algorithms import get_paths, read_file_routes, set_values_in_array
from utils import fitness_function, delete_file, simplify_path_list
from numpy.random import RandomState
import numpy as np
from random import randint, shuffle, uniform
from joblib import Parallel, delayed, cpu_count # --
import copy
from utils import Path, Pad
import time # pentru monitorizare timpilor, va fi necesar pentru stabilirea numarului de core-uri din joblib

# Genetic Algorithms param
POPULATION_SIZE:    int
ROUNDS:             int
MUTATIONS:          int
CROSSOVERS_1P:      int
CROSSOVERS_2P:      int
PARENTS_KEPT:       int
MAX_JOBS:           int
INIT_JOBS:          int
CROSSOVER_1P_JOBS:  int
CROSSOVER_2P_JOBS:  int
MUTATION_JOBS:      int
NEW_INDIVIDUALS:    int
NEW_INDIV_JOBS:     int

# File specific parameters
AXIS_MULT = None
ROWS:               int
COLUMNS:            int
OFFSET_X = 0
OFFSET_Y = 0


pad_list = []
width_list = []
clearance_list = []


# Variables used for finding the best solution
population = []
template_grid = None
routes = [] # global to ensure that every function has access to it
nr_needed_routes = 0 # how to ignore the routes that are defined but not used


# maybe i will insert in mutation / crossover a way to move a cell to adjent one in order to reduce some angles
class Individual:
        # --- de adaugat o metoda de verificat d 
        def __init__(self, order = [], grid = [], paths: Path = None, unplaced_routes_number = 0, paths_total_cost = -1):
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
    rng = RandomState()         # Inițializare cu sămânță implicită
    rng.shuffle(order_list)
    #shuffle(order_list)
    return order_list


# return a list sorted according to another one containing indexes
def order_list(original_list, indexes_list:list):
    modified_list = copy.deepcopy(original_list)
    for i in range(len(indexes_list)):
        index = indexes_list[i]
        modified_list[i], modified_list[index] = modified_list[index], modified_list[i]
    return modified_list


# return data needed for Individual type object (order of routing - list, routes placed - array, grid with areas used / blocked)
    # TODO routes number will be used to either remove cycles or will be dropped
def generate_individual(ordered_index_routes = None, routes_number = 0, starting_grid = None, existing_paths = None): # maybe add option for a starting grid and paths
    global routes, nr_needed_routes
    if ordered_index_routes:
        random_order_indexes = ordered_index_routes
    else:
        random_order_indexes = random_order_list(nr_needed_routes)

    ordered_routes = order_list(original_list=routes, indexes_list = random_order_indexes)

    global template_grid, ROWS, COLUMNS, pad_list, clearance_list, width_list

    if starting_grid:
        grid, possible_solution = get_paths(grid = starting_grid, routes = ordered_routes, pads = pad_list,
                                            existing_paths = existing_paths,
                                            width_list = width_list, clearance_list = clearance_list)
        # grid, possible_solution = get_paths(grid = starting_grid, routes = ordered_routes, pins_sizes = 1,
        #                                                 rows = ROWS, columns = COLUMNS)
    else:
        grid, possible_solution = get_paths(grid = template_grid, routes = ordered_routes, pads = pad_list,
                                            existing_paths = existing_paths,
                                            width_list = width_list, clearance_list = clearance_list)
    # possible_solution = list(Path)
    print(" ", grid[1][1], random_order_indexes)
    return Individual(order = random_order_indexes, 
                      paths = possible_solution, grid = grid)  


# for paralelization
@delayed
def parallel_generate_individual(routes_number):
    value = generate_individual(routes_number = routes_number)
    #print(value.order)
    return value


# initialize population; used for first generation
def parallel_population_generation(population_size: int, routes_number, n_jobs: int):
    global template_grid, routes
    #population = []
    #for i in range(population_size):
    #    population.append(generate_individual(routes_number=routes_number))
    population = Parallel(n_jobs = n_jobs)(
        parallel_generate_individual(routes_number) for _ in range(population_size)
    )
    return population


# return a value in range 0 - (n-1) that indicates how many elements are in common of the same index; 
# return value: 0 - nothing in common, n-1 - everything
def common_part_index(list1, list2):
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return i
    return len(list1)


# operations used for crossover; it is applied directly onto the children
def crossover_index_lists(child1_order, child2_order, crossover_points = 1):
    rng = RandomState()
    size = len(child1_order)
    index = rng.randint(0, size - 1)

    for _ in range(crossover_points):
        aux1_order = child1_order
        aux2_order = child2_order
        child1_order = aux1_order[:index] + aux2_order[index:]
        child2_order = aux2_order[:index] + aux1_order[index:]
        
        if len(child1_order) != len(set(child1_order)):
            return [], []

    return child1_order, child2_order


# max 5 tries to find 2 suitable parents for a 1 point crossover else returns the parents values; returns 2 children
def crossover(crossover_points = 1):
    global population, POPULATION_SIZE
    rng = RandomState()
    index1 = rng.randint(0, POPULATION_SIZE-1)
    index2 = rng.randint(0, POPULATION_SIZE-1)

    tries = 5
    while tries > 0:
        tries -= 1
        while index1 == index2:
            index2 = rng.randint(0, POPULATION_SIZE-1)
        
        parent1 = population[index1]
        parent2 = population[index2]

        parent_order1 = parent1.order
        parent_order2 = parent2.order
        #child1 = copy.deepcopy(parent1)
        #child2 = copy.deepcopy(parent2)
        child_order1, child_order2 = crossover_index_lists(parent_order1, parent_order2, crossover_points)
        if child_order1 == [] or child_order2 == []:
            child1 = generate_individual(ordered_index_routes = child_order1)
            child2 = generate_individual(ordered_index_routes = child_order2)
            return child1, child2

        index1 = rng.randint(0, POPULATION_SIZE-1)
        index2 = rng.randint(0, POPULATION_SIZE-1)

    return copy.deepcopy(population[index1]), copy.deepcopy(population[index2])


# perform a mutation on a individual and return the child
def mutation(parent: Individual):       # later can be modified to replace one segment from a circuit to another (3 or more points routing)
    order = copy.deepcopy(parent.order)
    size = len(order)
    rng = RandomState()
    index1 = rng.randint(0, size-1)
    index2 = rng.randint(0, size-1)

    while index1 == index2:
        index2 = rng.randint(0, size-1)
    
    order[index1], order[index2] = order[index1], order[index2] # swap the order for routing

    # to add a starting matrix from 0 to index
    child = generate_individual(ordered_index_routes=order)

    return child


# Funcția pentru mutație, paralelizată cu joblib, folosind o selecție aleatorie a indivizilor
def parallel_mutation(population, count):
    # Selectează indivizii aleatori din populație
    global MUTATION_JOBS
    rng = RandomState()
    selected_population = [rng.randint(0, len(population)-1) for _ in range(count)]
    #print(selected_population)
    #for i in range(len(selected_population)):
    #    print(population[i].order)

    # Paralelizarea mutației pentru indivizii selectați
    mutation_results = Parallel(n_jobs = MUTATION_JOBS)(
        delayed(mutation)(population[i]) for i in selected_population
    )
    return mutation_results



# Selection based on roulette wheel; returns list on indexes of individuals selected from a generation
def roulette_wheel_selection(population, selected_number):
    # Sum of all fitness values of individuals
    sum_fitness = sum(individual.total_cost for individual in population)

    def select_individual(selection_point):
        sum_selection = 0
        index = 0
        for individual in population:
            sum_selection += individual.total_cost
            if sum_selection >= selection_point:
                return index
            index += 1

    selection_points = [uniform(0, sum_fitness) for _ in range(selected_number)]
    selected = Parallel(n_jobs=-1)(delayed(select_individual)(point) for point in selection_points)
    #print(selected)
    return selected


# Adjusts the parameters (population size, number of generations, crossover and mutation operations per generatiosn, parents kept
# read the file -- find size of the matrix used for representation, offset on X, Y, axis multiplying factor, areas to be avoided
def set_parameters_GA(file_name: str, population_coef = 2, rounds_coef = 3, 
                      parents_coef = 35, crossover_1P_coef = 10, crossover_2P_coef = 10, mutations_coef = 5):
    global routes, nr_needed_routes, pad_list, width_list, clearance_list
    routes, colors = read_file_routes(file_name=file_name, draw = False)
    nr_needed_routes = len(routes)
    blocked_areas = None


    # TODO
    width_list = [1 for i in range(len(routes))]
    clearance_list = [1 for i in range(len(routes))]
    for route in routes:
        pin = Pad(center_x = route[0], center_y = route[1], length = 1, width = 1, occupied_area = [(route[0], route[1])])
        if pin not in pad_list:
            pad_list.append(pin)
        pin = Pad(center_x = route[2], center_y = route[3], length = 1, width = 1, occupied_area = [(route[2], route[3])])
        if pin not in pad_list:
            pad_list.append(pin)
    #

    # from file find x y coordinates for grid, if there are no such values, extract from min max
    global ROWS, COLUMNS, OFFSET_X, OFFSET_Y, AXIS_MULT, MAX_JOBS
    OFFSET_X = 0
    OFFSET_Y = 0
    AXIS_MULT = 1
    ROWS = 55
    COLUMNS = 55
    MAX_JOBS = cpu_count() - 4

    global POPULATION_SIZE, INIT_JOBS, ROUNDS
    POPULATION_SIZE = min(int(nr_needed_routes * population_coef), 50) # 55
    INIT_JOBS = min(POPULATION_SIZE / 5 + 1, MAX_JOBS)     # GUI + genetic_routing 
    ROUNDS = min(int(nr_needed_routes * rounds_coef), 50) # 25

    global PARENTS_KEPT, CROSSOVERS_1P, CROSSOVERS_2P, CROSSOVER_1P_JOBS, CROSSOVER_2P_JOBS
    global MUTATIONS, MUTATION_JOBS, NEW_INDIVIDUALS, NEW_INDIV_JOBS
    PARENTS_KEPT = int((POPULATION_SIZE * parents_coef) / 100) # 25
    CROSSOVERS_1P = int((POPULATION_SIZE * crossover_1P_coef) / 100) # 5
    CROSSOVERS_2P = int((POPULATION_SIZE * crossover_2P_coef) / 100)
    CROSSOVER_1P_JOBS = min(CROSSOVERS_1P / 5 + 1, MAX_JOBS)
    CROSSOVER_2P_JOBS = min(CROSSOVERS_2P / 5 + 1, MAX_JOBS)
    MUTATIONS = int((nr_needed_routes * POPULATION_SIZE * mutations_coef) / 100) # 10
    MUTATION_JOBS = min(MUTATIONS / 5 + 1, MAX_JOBS)
    INIT_JOBS = min(MUTATIONS / 5 + 1, MAX_JOBS)     # GUI + genetic_routing 
    NEW_INDIVIDUALS = POPULATION_SIZE - PARENTS_KEPT - CROSSOVERS_1P * 2 - CROSSOVERS_2P * 2 - MUTATIONS
    NEW_INDIV_JOBS = min(NEW_INDIVIDUALS / 5 + 1, MAX_JOBS)

    global template_grid
    template_grid = np.zeros((ROWS, COLUMNS), dtype=int)
    template_grid = set_values_in_array(blocked_cells = blocked_areas, arr = template_grid, value = 0)

    #individual_routes_placed = [[[] for j in range(routes_number)] for i in range(POPULATION_SIZE)] - instead of individual class
    #individual_grid = [[[] for j in range(routes_number)] for i in range(POPULATION_SIZE)]



# function that performs the routing with GA based on the parameters from first_generation()
def genetic_routing(file_name, event = None):
    file1 = open(file_name, "a")

    global population, best_solution, nr_needed_routes, POPULATION_SIZE, ROUNDS, PARENTS_KEPT, \
        CROSSOVERS_1P, CROSSOVERS_2P, CROSSOVER_1P_JOBS, CROSSOVER_2P_JOBS, \
        MUTATIONS, MUTATION_JOBS, NEW_INDIVIDUALS, NEW_INDIV_JOBS
    for i in range(ROUNDS):
        if not event or not event.is_set():
            print(f"Generation {i+1}")
            for j in range(POPULATION_SIZE):
                if population[j].total_cost == -1:
                    paths_info = population[j].paths
                    paths = [x.path for x in paths_info] # Problema iterare path-uri
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
            #aux = best_solution.paths
            aux = [x.path for x in best_solution.paths]
            #best_paths = simplify_path_list(best_solution.paths)
            best_paths = simplify_path_list(paths_list = aux)
            if best_paths:
                file1.write(f"{best_paths}\n")

            next_generation = []

            indexes = roulette_wheel_selection(population=population, selected_number=PARENTS_KEPT)
            for index in indexes:
                next_generation.append(copy.deepcopy(population[index]))

            #for _ in range(0, CROSSOVERS_1P):
            #    child1, child2 = crossover(crossover_points = 1)
            #    next_generation.append(child1)
            #    next_generation.append(child2)

            children_crossover_1p = Parallel(n_jobs = CROSSOVER_1P_JOBS)(
                delayed(crossover)(1) for _ in range(CROSSOVERS_1P)
            )
            for child in children_crossover_1p:
                next_generation.extend(child)

            children_crossover_2p = Parallel(n_jobs = CROSSOVER_2P_JOBS)(
                delayed(crossover)(2) for _ in range(CROSSOVERS_2P)
            )
            for child in children_crossover_2p:
                next_generation.extend(child)

            #for _ in range(0, CROSSOVERS_2P):
            #    child1, child2 = crossover(crossover_points = 2)
            #    next_generation.append(child1)
            #    next_generation.append(child2) 
            
            #for j in range(0, MUTATIONS):
            #    index = randint(0, POPULATION_SIZE-1)
            #    child = mutation(population[index])
            #    next_generation.append(child)

            # Paralelizarea mutației
            mutated_children = parallel_mutation(population, MUTATIONS)
            next_generation.extend(mutated_children)
            
            # maybe add another type of mutation or crossover operators

            # add new individuals into the new generation

            new_individuals = parallel_population_generation(population_size = NEW_INDIVIDUALS,
                                                              routes_number = nr_needed_routes, n_jobs = NEW_INDIV_JOBS)
            next_generation.extend(new_individuals)
            #try:
            #    for j in range(count_new_individuals):
            #        individual = generate_individual()
            #        next_generation.append(individual)
            #except Exception as e:
            #    print("Error while adding new individuals; Error:", e) 
            
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
        global population, POPULATION_SIZE, nr_needed_routes, INIT_JOBS
        population = parallel_population_generation(population_size = POPULATION_SIZE, 
                                                    routes_number = nr_needed_routes, n_jobs = INIT_JOBS)

        #print("Done")           # -- to check
        #for i in range(len(population)):
        #    print(population[i].order)
        
        genetic_routing(file_name = save_file, event = event)
        print(f"\n\norder = {best_solution.order}, cost = {best_solution.total_cost}, paths = {best_solution.paths}")
        
        if event:
            event.set()


if __name__ == "__main__":
    run_genetic_algorithm(save_file = "solution.txt", read_file = "pins.csv")
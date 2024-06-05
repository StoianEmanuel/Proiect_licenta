#! C:\Users\manue\Desktop\GA\Proiect\venv\Scripts\python.exe
from routing_algorithms import get_paths
from pcb_utils import get_data_for_GA, get_segments_for_board, write_segments_to_EOF
from utils import fitness_function, get_simplified_paths
from numpy.random import RandomState
from random import uniform
from joblib import Parallel, delayed, cpu_count # --
import copy
from utils import Path, Pad
import time # pentru monitorizare timpilor, va fi necesar pentru stabilirea numarului de core-uri din joblib

# Genetic Algorithms params
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


pads = []
planned_routes = []
netcodes_list  = []
nr_needed_routes = 0 # how to ignore the routes that are defined but not used

# Variables used for finding the best solution
population = []
template_grid = None
process_callback = None

# maybe i will insert in mutation / crossover a way to move a cell to adjent one in order to reduce some angles
class Individual:
        # --- de adaugat o metoda de verificat d 
        def __init__(self, order = [], paths: Path = None, unplaced_routes_number: int = 0, paths_total_cost = -1):
            self.order  = order    # holds order of routes from input
            self.paths  = paths    # [[path1], [path2], ...]
            self.unplaced_routes_number = unplaced_routes_number # from the necessary ones
            self.total_cost = paths_total_cost      # fitness_value; if no paths would be 0

        def __repr__(self):
            return repr((self.order, self.paths, self.unplaced_routes_number, self.total_cost))
        
        @staticmethod
        def return_path_cost(self):
            return self.total_cost

        def __str__(self) -> str:
            return f'Order: {self.order}; Total cost: {self.total_cost}'

best_solution = Individual()


# Returns a possible solution based on random order of netcode
def generate_individual(ordered_indexes = None, starting_grid = None):
    global nr_needed_routes, template_grid, ROWS, COLUMNS, pads, planned_routes, netcodes_list

    if not ordered_indexes:
        rng = RandomState()         # Inițializare cu sămânță implicită
        random_order_indexes = copy.copy(netcodes_list)
        rng.shuffle(random_order_indexes)
    else:
        random_order_indexes = ordered_indexes

    if starting_grid:
        possible_solution = get_paths(starting_grid, (ROWS, COLUMNS), planned_routes,
                                        random_order_indexes, pads)
    else:
        possible_solution = get_paths(template_grid, (ROWS, COLUMNS), planned_routes, 
                                        random_order_indexes, pads)
    # possible_solution = list(Path)

    return Individual(order = random_order_indexes, paths = possible_solution)  


# for paralelization
@delayed
def parallel_generate_individual():
    value = generate_individual()
    return value


# initialize population; used for first generation
def parallel_population_initialize(population_size: int, n_jobs: int):
    population = Parallel(n_jobs = n_jobs)(
        parallel_generate_individual() for _ in range(population_size)
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
            child1 = generate_individual(child_order1)
            child2 = generate_individual(child_order2)
            return child1, child2

        index1 = rng.randint(0, POPULATION_SIZE-1)
        index2 = rng.randint(0, POPULATION_SIZE-1)

    return copy.deepcopy(population[index1]), copy.deepcopy(population[index2])


# perform a mutation on a individual and return the child
def mutation(parent: Individual):       # later can be modified to replace one segment from a circuit to another (3 or more points routing)
    order = copy.deepcopy(parent.order)
    global nr_needed_routes
    size = nr_needed_routes
    rng = RandomState()
    index1 = rng.randint(0, size-1)
    index2 = rng.randint(0, size-1)

    while index1 == index2:
        index2 = rng.randint(0, size-1)
    
    order[index1], order[index2] = order[index1], order[index2] # swap the order for routing

    # to add a starting matrix from 0 to index
    child = generate_individual(order)

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

    global MAX_JOBS
    selection_points = [uniform(0, sum_fitness) for _ in range(selected_number)]
    selected = Parallel(n_jobs=MAX_JOBS)(delayed(select_individual)(point) for point in selection_points)
    return selected


# Adjusts the parameters (population size, number of generations, crossover and mutation operations per generatiosn, parents kept
# New one --- TODO
def set_parameters_GA(population_coef = 2, rounds_coef = 3, parents_coef = 35, 
                      crossover_1P_coef = 10, crossover_2P_coef = 10, mutations_coef = 5):
    global MAX_JOBS, nr_needed_routes, POPULATION_SIZE, INIT_JOBS, ROUNDS
    MAX_JOBS = max(1, cpu_count() - 4)
    POPULATION_SIZE = min(int(nr_needed_routes * population_coef), 25) # 55
    INIT_JOBS = min(int(POPULATION_SIZE / 5 + 1), MAX_JOBS)     # GUI + genetic_routing 
    ROUNDS = min(int(nr_needed_routes * rounds_coef), 25) # 25

    global PARENTS_KEPT, CROSSOVERS_1P, CROSSOVERS_2P, CROSSOVER_1P_JOBS, CROSSOVER_2P_JOBS
    PARENTS_KEPT = int((POPULATION_SIZE * parents_coef) / 100) # 25
    CROSSOVERS_1P = int((POPULATION_SIZE * crossover_1P_coef) / 100) # 5
    CROSSOVERS_2P = int((POPULATION_SIZE * crossover_2P_coef) / 100)
    CROSSOVER_1P_JOBS = min(CROSSOVERS_1P / 5 + 1, MAX_JOBS)
    CROSSOVER_2P_JOBS = min(CROSSOVERS_2P / 5 + 1, MAX_JOBS)

    global MUTATIONS, MUTATION_JOBS, NEW_INDIVIDUALS, NEW_INDIV_JOBS   
    MUTATIONS = int((nr_needed_routes * POPULATION_SIZE * mutations_coef) / 100) # 10
    MUTATION_JOBS = min(MUTATIONS / 5 + 1, MAX_JOBS)

    NEW_INDIVIDUALS = POPULATION_SIZE - PARENTS_KEPT - CROSSOVERS_1P * 2 - CROSSOVERS_2P * 2 - MUTATIONS
    NEW_INDIV_JOBS = min(NEW_INDIVIDUALS / 5 + 1, MAX_JOBS)



# New one ----------- TODO
def genetic_routing(filename, logs_file, event = None):
    file1 = open(logs_file, "a") # Se va renunta la file open

    global process_callback, population, best_solution, POPULATION_SIZE, ROUNDS, PARENTS_KEPT, \
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
                    cost = fitness_function(paths, n, 2)
                    population[j].total_cost = cost
                
            population.sort(reverse = False, key = Individual.return_path_cost)
        
            if best_solution.total_cost == -1 or \
                (population[0].total_cost != -1 and population[0].total_cost < best_solution.total_cost):
                best_solution = copy.deepcopy(population[0])

            file1.write(f"{i}: {str(best_solution)}\n")
            #aux = best_solution.paths
            aux = [x.path for x in best_solution.paths]
            best_paths = get_simplified_paths(paths_list = aux)

            if best_paths:
                file1.write(f"{best_paths}\n")

            next_generation = []

            indexes = roulette_wheel_selection(population, PARENTS_KEPT)
            for index in indexes:
                next_generation.append(copy.deepcopy(population[index]))

            children_crossover_1p = Parallel(CROSSOVER_1P_JOBS)(
                delayed(crossover)(1) for _ in range(CROSSOVERS_1P)
            )
            for child in children_crossover_1p:
                next_generation.extend(child)

            children_crossover_2p = Parallel(CROSSOVER_2P_JOBS)(
                delayed(crossover)(2) for _ in range(CROSSOVERS_2P)
            )
            for child in children_crossover_2p:
                next_generation.extend(child)

            # Paralelizarea mutației
            mutated_children = parallel_mutation(population, MUTATIONS)
            next_generation.extend(mutated_children)
            
            # maybe add another type of mutation or crossover operators

            # add new individuals into the new generation
            new_individuals = parallel_population_initialize(NEW_INDIVIDUALS, NEW_INDIV_JOBS)
            next_generation.extend(new_individuals)
            
            #for j in range(POPULATION_SIZE):   file1.write(f"{i}, {j}, {next_generation[j].order}, {next_generation[j].total_cost}\n")
            population = next_generation
            process_callback = (ROUNDS, i + 1)


    global OFFSET_X, OFFSET_Y, AXIS_MULT, planned_routes
    best_paths = best_solution.paths
    for index in range(len(netcodes_list)):
        best_paths[index].update_simplified_path()

    segments = get_segments_for_board(best_paths, planned_routes, (OFFSET_Y, OFFSET_X), AXIS_MULT)

    write_segments_to_EOF(filename, segments)
    
    file1.close()


# New one ----------------------- # TODO
def run_genetic_algorithm(filename, event = None, process_callback = None, **kwargs):
    if not event or not event.is_set():
        global template_grid, pads, planned_routes, netcodes_list, OFFSET_Y, OFFSET_X, AXIS_MULT
        template_grid, AXIS_MULT, OFFSET_Y, OFFSET_X, pads, planned_routes, netcodes_list = get_data_for_GA(filename, **kwargs)

        global ROWS, COLUMNS, nr_needed_routes
        ROWS, COLUMNS = len(template_grid), len(template_grid[0])
        nr_needed_routes = len(netcodes_list)


        set_parameters_GA()

        print("Populare ...")
        global population, POPULATION_SIZE, INIT_JOBS
        print("Params", POPULATION_SIZE, INIT_JOBS, MAX_JOBS, MUTATIONS, CROSSOVERS_1P, CROSSOVERS_2P)

        population = parallel_population_initialize(POPULATION_SIZE, INIT_JOBS)
        print("Populare finalizata")

        process_callback = process_callback

        genetic_routing(filename, logs_file= "a.txt", event = event)

        print(f"\n\norder = {best_solution.order}, cost = {best_solution.total_cost}, paths = {best_solution.paths}")

        if event:
            event.set()

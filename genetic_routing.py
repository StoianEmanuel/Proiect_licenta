from routing_algorithms import get_paths
from pcb_utils import get_data_for_GA, get_segments_for_board, write_segments_to_EOF
from utils import fitness_function, get_simplified_paths, get_perpendicular_direction, \
                    mark_path_in_array, update_grid_with_paths, check_90_deg_bend
from routing_algorithms import a_star_search
from numpy.random import RandomState
from random import uniform
from joblib import Parallel, delayed, cpu_count # --
import copy
from utils import Path
import time # pentru monitorizare timpilor, va fi necesar pentru stabilirea numarului de core-uri din joblib



class Individual:
        def __init__(self, order = [], paths: Path = None, unplaced_routes_number: int = 0, paths_total_cost = -1):
            self.order  = order    # holds order of routes from input
            self.paths  = paths    # [[path1], [path2], ...]
            self.unplaced_routes_number = unplaced_routes_number # from the necessary ones
            self.total_cost = paths_total_cost      # fitness_value

        def __repr__(self):
            return repr((self.order, self.paths, self.unplaced_routes_number, self.total_cost))
        
        @staticmethod
        def return_path_cost(self):
            return self.total_cost

        def __str__(self) -> str:
            return f'Order: {self.order}; Total cost: {self.total_cost}'


# Genetic Algorithms params
POPULATION_SIZE:    int
ROUNDS:             int
MUTATIONS_ORDER:    int
MUTATIONS_PATH:     int
CROSSOVERS_1P:      int
CROSSOVERS_2P:      int
PARENTS_KEPT:       int
MAX_JOBS:           int
INIT_JOBS:          int
CROSSOVER_1P_JOBS:  int
CROSSOVER_2P_JOBS:  int
MUTATION_ORDER_JOBS:int
MUTATION_PATH_JOBS: int
NEW_INDIVIDUALS:    int
NEW_INDIV_JOBS:     int

# Scaling and offset
AXIS_MULT = None
ROWS:               int
COLUMNS:            int
LAYERS:             int
OFFSET_X = 0
OFFSET_Y = 0

# Routes proprieties
planned_routes = []
netcodes_list  = []
NR_PATH_PLANNED = 0
pads_list = []
template_grid = None

# GA population
population = []
best_solution = Individual()


# Returns a possible solution based on random order of netcode
def generate_individual(ordered_indexes = None, starting_grid = None):
    global NR_PATH_PLANNED, template_grid, LAYERS, ROWS, COLUMNS, pads_list, planned_routes, netcodes_list

    if not ordered_indexes:
        rng = RandomState()
        random_order_indexes = copy.copy(netcodes_list)
        rng.shuffle(random_order_indexes)
    else:
        random_order_indexes = ordered_indexes

    if starting_grid:
        possible_solution = get_paths(starting_grid, (LAYERS, ROWS, COLUMNS), planned_routes,
                                        random_order_indexes, pads_list)
    else:
        possible_solution = get_paths(template_grid, (LAYERS, ROWS, COLUMNS), planned_routes, 
                                        random_order_indexes, pads_list)

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
def path_mutation(parent: Individual):    
    order = copy.deepcopy(parent.order)
    global NR_PATH_PLANNED, template_grid, ROWS, COLUMNS, LAYERS

    rng = RandomState()
    path_index = rng.randint(0, NR_PATH_PLANNED-1)
    mutated_path = parent.paths[path_index]

    if mutated_path.path:
        current_path = mutated_path.path
        length = len(current_path)
        if length < 10:
            return parent
        
        grid = copy.deepcopy(template_grid)
        previous_paths = parent.paths[0 : path_index]
        update_grid_with_paths(grid, (ROWS, COLUMNS), previous_paths)

        layer = mutated_path.layer_id
        width = mutated_path.width
        clearance = mutated_path.clearance
        netcode = mutated_path.netcode

        deviation = rng.randint(1, 5)
        deviation_sign = rng.choice([-1, 1])

        start_i = rng.randint(3, length-6)
        end_i = rng.randint(start_i + 2, length - 3)
        
        (start_y_aux1, start_x_aux1), (start_y_aux2, start_x_aux2) = current_path[start_i-2], current_path[start_i-1]
        start_y, start_x = current_path[start_i]
        
        (end_y_aux1, end_x_aux1), (end_y_aux2, end_x_aux2) = current_path[end_i+2], current_path[end_i+1]
        end_y, end_x = current_path[end_i]

        dir_y1, dir_y2, dir_y3, dir_y4 = start_y_aux1 - start_y_aux2, start_y_aux2 - start_y, end_y_aux1 - end_y, end_y_aux2 - end_y_aux1
        dir_x1, dir_x2, dir_x3, dir_x4 = start_x_aux1 - start_x_aux2, start_x_aux2 - start_x, end_x_aux1 - end_x, end_x_aux2 - end_x_aux1

        # start
        if not check_90_deg_bend((dir_y1, dir_x1), (dir_y2, dir_x2)):
            dir_y_start, dir_x_start = start_y_aux2 - start_y_aux1, start_x_aux2 - start_x_aux1
        else:   
            dir_y_start, dir_x_start = start_y - start_y_aux1, start_x - start_x_aux1
            dir_perp_y_start, dir_perp_x_start = get_perpendicular_direction(dir_y_start, dir_x_start)
            dir_y_start, dir_x_start = dir_perp_y_start, dir_perp_x_start

        # end
        if not check_90_deg_bend((dir_y3, dir_x3), (dir_y4, dir_x4)):
            dir_y_end, dir_x_end = end_y_aux2 - end_y_aux1, end_x_aux2 - end_x_aux1
        else:   
            dir_y_end, dir_x_end = end_y - end_y_aux2, end_x - end_x_aux2
            dir_perp_y_end, dir_perp_x_end = get_perpendicular_direction(dir_y_end, dir_x_end)
            dir_y_end, dir_x_end = dir_perp_y_end, dir_perp_x_end


        grid[layer][start_y][start_x] = -1
        grid[layer][end_y][end_x] = -1


        start_list = []
        end_list   = []

        for j in range(deviation):
            p1_y, p1_x = start_y_aux1 + j * deviation_sign * dir_y_start, start_x_aux1 + j * deviation_sign * dir_x_start
            p2_y, p2_x = end_y_aux2   + j * deviation_sign * dir_y_start, end_x_aux2   + j * deviation_sign * dir_x_start

            if grid[layer][p1_y][p1_x] == 0 or grid[layer][p2_y][p2_x] == 0:
               grid[layer][p1_y][p1_x] = -1
               grid[layer][p2_y][p2_x] = -1
            else:
                p1_y, p1_x = p1_y - deviation_sign * dir_y_start, p1_x - deviation_sign * dir_x_start
                p2_y, p1_x = p2_y - deviation_sign * dir_y_start, p2_x - deviation_sign * dir_x_start
                break

            start_list.append((p1_y, p1_x))
            end_list.append((p2_y, p2_x))
        
        grid[layer][p1_y][p1_x] = 0
        grid[layer][p2_y][p2_x] = 0
        deviation_path = a_star_search(grid[layer][:][:], (ROWS, COLUMNS), (p1_y, p1_x), (p2_y, p2_x), netcode, clearance, width)
        if not deviation_path:
            return parent

        current_path = current_path[:start_i] + start_list + deviation_path + end_list + current_path[end_i+1:]
        grid[layer][:][:] = mark_path_in_array(grid[layer][:][:], current_path, netcode)
        updated_path = Path(mutated_path.start, mutated_path.destination, netcode, current_path, width, clearance, True, None, layer)
        
        aux_planned_routes = copy.deepcopy(planned_routes)
        for path_object in (previous_paths + [updated_path]):
            if path_object:
                netcode = path_object.netcode
                aux_planned_routes[netcode].existing_conn.append((path_object.start[0], path_object.start[1],
                                                                   path_object.destination[0], path_object.destination[1]))

        slice_order = order[path_index+1:]
        new_paths = get_paths(grid, (LAYERS, ROWS, COLUMNS), aux_planned_routes, slice_order, pads_list)

        child = Individual(order, previous_paths + [updated_path] + new_paths)
        return child
    
    return parent


# Funcția pentru mutație, paralelizată cu joblib, folosind o selecție aleatorie a indivizilor
def parallel_path_mutation(population, count):
    # Selectează indivizii aleatori din populație
    global MUTATION_PATH_JOBS
    rng = RandomState()
    selected_population = [rng.randint(0, len(population)-1) for _ in range(count)]

    # Paralelizarea mutației pentru indivizii selectați
    mutation_results = Parallel(n_jobs = MUTATION_ORDER_JOBS)(
        delayed(path_mutation)(population[i]) for i in selected_population
    )
    return mutation_results



# perform a mutation on a individual and return the child
def order_mutation(parent: Individual):       # later can be modified to replace one segment from a circuit to another (3 or more points routing)
    order = copy.deepcopy(parent.order)
    global NR_PATH_PLANNED
    rng = RandomState()
    index1 = rng.randint(0, NR_PATH_PLANNED-1)
    index2 = rng.randint(0, NR_PATH_PLANNED-1)

    while index1 == index2:
        index2 = rng.randint(0, NR_PATH_PLANNED-1)
    
    order[index1], order[index2] = order[index1], order[index2] # swap the order for routing

    # to add a starting matrix from 0 to index
    child = generate_individual(order)

    return child


# Funcția pentru mutație, paralelizată cu joblib, folosind o selecție aleatorie a indivizilor
def parallel_order_mutation(population, count):
    # Selectează indivizii aleatori din populație
    global MUTATION_ORDER_JOBS
    rng = RandomState()
    selected_population = [rng.randint(0, len(population)-1) for _ in range(count)]

    # Paralelizarea mutației pentru indivizii selectați
    mutation_results = Parallel(n_jobs = MUTATION_ORDER_JOBS)(
        delayed(order_mutation)(population[i]) for i in selected_population
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
                      crossover_1P_coef = 10, crossover_2P_coef = 10, mutation_order_coef = 5, mutation_path_coef = 10): # 5
    global MAX_JOBS, NR_PATH_PLANNED, POPULATION_SIZE, INIT_JOBS, ROUNDS
    MAX_JOBS = max(1, cpu_count() - 4)
    POPULATION_SIZE = min(int(NR_PATH_PLANNED * population_coef), 25) # 55
    INIT_JOBS = min(int(POPULATION_SIZE / 5 + 1), MAX_JOBS)     # GUI + genetic_routing 
    ROUNDS = min(int(NR_PATH_PLANNED * rounds_coef), 25) # 25

    global PARENTS_KEPT, CROSSOVERS_1P, CROSSOVERS_2P, CROSSOVER_1P_JOBS, CROSSOVER_2P_JOBS
    PARENTS_KEPT = int((POPULATION_SIZE * parents_coef) / 100) # 25
    CROSSOVERS_1P = int((POPULATION_SIZE * crossover_1P_coef) / 100) # 5
    CROSSOVERS_2P = int((POPULATION_SIZE * crossover_2P_coef) / 100)
    CROSSOVER_1P_JOBS = min(CROSSOVERS_1P / 5 + 1, MAX_JOBS)
    CROSSOVER_2P_JOBS = min(CROSSOVERS_2P / 5 + 1, MAX_JOBS)

    global MUTATIONS_ORDER, MUTATIONS_PATH, MUTATION_ORDER_JOBS, MUTATION_PATH_JOBS, NEW_INDIVIDUALS, NEW_INDIV_JOBS   
    MUTATIONS_ORDER = int((NR_PATH_PLANNED * POPULATION_SIZE * mutation_order_coef) / 100) # 10
    MUTATION_ORDER_JOBS = min(MUTATIONS_ORDER / 5 + 1, MAX_JOBS)
    MUTATIONS_PATH = min(1, int((NR_PATH_PLANNED * POPULATION_SIZE * mutation_path_coef) / 100)) # 10
    MUTATION_PATH_JOBS = min(MUTATIONS_ORDER / 5 + 1, MAX_JOBS)

    NEW_INDIVIDUALS = POPULATION_SIZE - PARENTS_KEPT - CROSSOVERS_1P * 2 - CROSSOVERS_2P * 2 - MUTATIONS_ORDER - MUTATIONS_PATH
    NEW_INDIV_JOBS = min(NEW_INDIVIDUALS / 5 + 1, MAX_JOBS)


# New one ----------- TODO
def genetic_routing(filename, logs_file, event = None):
    file1 = open(logs_file, "a") # Se va renunta la file open

    global population, best_solution, POPULATION_SIZE, ROUNDS, PARENTS_KEPT, \
        CROSSOVERS_1P, CROSSOVERS_2P, CROSSOVER_1P_JOBS, CROSSOVER_2P_JOBS, \
        MUTATIONS_ORDER, MUTATION_ORDER_JOBS, MUTATIONS_PATH, MUTATION_PATH_JOBS, \
        NEW_INDIVIDUALS, NEW_INDIV_JOBS
    
    for i in range(ROUNDS):
        if not event or not event.is_set():
            print(f"Generation {i+1}")
            for j in range(POPULATION_SIZE):
                if population[j].total_cost == -1:
                    paths_info = population[j].paths
                    paths = [x.path for x in paths_info]
                    n = population[j].unplaced_routes_number
                    cost = fitness_function(paths, n, 2)
                    population[j].total_cost = cost
                
            population.sort(reverse = False, key = Individual.return_path_cost)
        
            if best_solution.total_cost == -1 or \
                (population[0].total_cost != -1 and population[0].total_cost < best_solution.total_cost):
                best_solution = copy.deepcopy(population[0])

            file1.write(f"{i}: {str(best_solution)}\n")

            aux = [x.path for x in best_solution.paths]
            best_paths = get_simplified_paths(paths_list = aux)

            if best_paths:
                file1.write(f"{best_paths}\n")

            next_generation = []

            # Pastrarea indivizilor prin roulette_wheel_selection
            indexes = roulette_wheel_selection(population, PARENTS_KEPT)
            for index in indexes:
                next_generation.append(copy.deepcopy(population[index]))

            # Paralelizare crossover
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
            mutated_children = parallel_path_mutation(population, MUTATIONS_PATH)
            next_generation.extend(mutated_children)

            mutated_children = parallel_order_mutation(population, MUTATIONS_ORDER)
            next_generation.extend(mutated_children)

            # Add new individuals into the new generation
            new_individuals = parallel_population_initialize(NEW_INDIVIDUALS, NEW_INDIV_JOBS)
            next_generation.extend(new_individuals)
            
            #for j in range(POPULATION_SIZE):   file1.write(f"{i}, {j}, {next_generation[j].order}, {next_generation[j].total_cost}\n")
            population = next_generation
   
    file1.close()


def update_board_file(filename, event):
    # Salvare rezultate in fisier
    global OFFSET_X, OFFSET_Y, AXIS_MULT, planned_routes
    if best_solution and (not event or not event.is_set()):
        best_paths = best_solution.paths
        for index in range(len(netcodes_list)):
            best_paths[index].update_simplified_path()

        segments = get_segments_for_board(best_paths, planned_routes, (OFFSET_Y, OFFSET_X), AXIS_MULT)

        write_segments_to_EOF(filename, segments)



# TODO : TODO add route deviations, costul rutelor existente - nu prea este relevant deoarece nu as tine cont de el
def run_genetic_algorithm(filename, event = None, **kwargs):
    if not event or not event.is_set():
        global template_grid, pads_list, planned_routes, netcodes_list, OFFSET_Y, OFFSET_X, AXIS_MULT, LAYERS
        template_grid, AXIS_MULT, OFFSET_Y, OFFSET_X, pads_list, planned_routes, netcodes_list, LAYERS = get_data_for_GA(filename, **kwargs)

        global ROWS, COLUMNS, NR_PATH_PLANNED
        ROWS, COLUMNS = len(template_grid[0]), len(template_grid[0][0])
        NR_PATH_PLANNED = len(netcodes_list)


        set_parameters_GA()

        print("Populare ...")
        global population, POPULATION_SIZE, INIT_JOBS
        print("Params", POPULATION_SIZE, INIT_JOBS, MAX_JOBS, MUTATIONS_ORDER, CROSSOVERS_1P, CROSSOVERS_2P)

        population = parallel_population_initialize(POPULATION_SIZE, INIT_JOBS)
        print("Populare finalizata")

        genetic_routing(filename, logs_file= "a.txt", event = event)

        update_board_file(filename, event)

        print(f"\n\norder = {best_solution.order}, cost = {best_solution.total_cost}, paths = {best_solution.paths}")

        if event:
            event.set()

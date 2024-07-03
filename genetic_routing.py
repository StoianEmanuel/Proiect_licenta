from pcb_utils import get_data_for_GA, get_segments_for_board, write_segments_to_EOF
from utils import Individual, fitness_function, get_simplified_paths
from random import uniform, sample
import random
from joblib import cpu_count # --
import copy
import os
from math import ceil
import time
from pathos.threading import ThreadPool
from genetic_operators import generate_individual, crossover, path_mutation, order_mutation
from multiprocessing import Pool

# Genetic Algorithms params
POPULATION_SIZE:    int
ROUNDS:             int
MUTATIONS_ORDER:    int
MUTATIONS_PATH:     int
CROSSOVERS:         int
CROSSOVERS_POINTS:  int
PARENTS_KEPT:       int
MAX_JOBS:           int
INIT_JOBS:          int
CROSSOVER_JOBS:     int
MUTATION_ORDER_JOBS:int
MUTATION_PATH_JOBS: int
NEW_INDIVIDUALS:    int
NEW_INDIV_JOBS:     int
SEED:               int
SELECTION:          bool

# Scaling and offset
AXIS_MULT        = None
ROWS:               int
COLUMNS:            int
LAYERS:             int
OFFSET_X:           int
OFFSET_Y:           int

# Routes proprieties
NR_PATH_PLANNED:    int
planned_routes  = []
netcodes_list   = []
pads_list       = []
template_grid   = None

# GA population
population      = []
best_solution   = Individual()


def generate_seeds(numbers, base_seed):
    random.seed(base_seed)
    return [random.random() for _ in range(numbers)]

# Extrack values from tuple and keep in a list 
def flatten_list(original_list):
    flattened_list = []
    for item in original_list:
        if isinstance(item, tuple):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    return flattened_list


def parallel_population_initialize(population_size: int, n_jobs: int, base_seed, ordered_indexes=None, starting_grid=None, template_grid=None, planned_routes=None, pads_list=None, netcodes_list=None, layers=None, rows=None, columns=None):
    seeds = generate_seeds(population_size, base_seed)
    args = [(seed, ordered_indexes, starting_grid, template_grid, planned_routes, pads_list, netcodes_list, layers, rows, columns) for seed in seeds]

    with Pool(n_jobs) as pool:
        population = pool.map(generate_individual, args)

    return population


def generate_crossover_args(seed, count, population, template_grid, planned_routes, pads_list, LAYERS, ROWS, COLUMNS):
    crossover_args = []
    random.seed(seed)
    for _ in range(count):
        parent1 = copy.deepcopy(random.choice(population))
        parent2 = copy.deepcopy(random.choice(population))
        crossover_args.append((random.random(), CROSSOVERS_POINTS, (parent1, parent2, template_grid, planned_routes, pads_list, LAYERS, ROWS, COLUMNS)))
    return crossover_args



def mutate_with_seed_path(idx, population, args):
    return path_mutation(population[idx], args)



def mutate_with_seed_order(idx, population, selected_indices, seeds, args):
    individual = population[idx]
    seed_idx = selected_indices.index(idx)
    seed = seeds[seed_idx]
    return order_mutation(seed, individual, args)


# Selection based on roulette wheel; returns list on indexes of individuals selected from a generation
def select_individual(args):
    individual, selection_point = args
    sum_selection = 0
    index = 0
    for ind in individual:
        sum_selection += ind.total_cost
        if sum_selection >= selection_point:
            return index
        index += 1



def parallel_genetic_operators(seed, crossover_count, mutation_path_count, mutation_order_count, population, args):
    NR_PATH_PLANNED, template_grid, planned_routes, pads_list, netcodes_list, LAYERS, ROWS, COLUMNS = args

    random.seed(seed)
    seed1 = random.random()
    crossover_args = generate_crossover_args(seed1, crossover_count, population, template_grid, planned_routes, pads_list, LAYERS, ROWS, COLUMNS)


    args1 = (NR_PATH_PLANNED, template_grid, planned_routes, pads_list, LAYERS, ROWS, COLUMNS)
    parents_path_mutation = [random.randint(0, len(population)-1) for _ in range(mutation_path_count)]


    args2 = (NR_PATH_PLANNED, template_grid, planned_routes, pads_list, netcodes_list, LAYERS, ROWS, COLUMNS)
    parents_order_mutation = random.sample(range(len(population)), mutation_order_count)
    seeds = [random.random() for _ in range(mutation_order_count)]

    n_jobs = min(crossover_count + mutation_path_count + mutation_order_count, MAX_JOBS)
    with Pool(n_jobs) as pool:
        crossover_population = pool.starmap(crossover, crossover_args)
        mutated_path_population = pool.starmap(mutate_with_seed_path, [(idx, population, args1) for idx in parents_path_mutation])
        mutated_order_population = pool.starmap(mutate_with_seed_order, [(idx, population, parents_order_mutation, seeds, args2) for idx in parents_order_mutation])

    result = []
    result.extend(flatten_list(crossover_population))
    result.extend(flatten_list(mutated_path_population))
    result.extend(flatten_list(mutated_order_population))
    return result


def roulette_wheel_selection(population, selected_number):
    # Sum of all fitness values of individuals
    sum_fitness = sum(individual.total_cost for individual in population)

    global MAX_JOBS
    selection_points = [(population, uniform(0, sum_fitness)) for _ in range(selected_number)]
    with ThreadPool(MAX_JOBS) as pool:
        selected = pool.map(select_individual, selection_points)
    return selected


def tournament_selection(seed, population, tournament_size, num_parents):
    random.seed(seed)
    selected_parents = []
    for _ in range(num_parents):
        tournament = sample(range(len(population)), tournament_size)
        winner_index = min(tournament, key=lambda i: population[i].total_cost)
        selected_parents.append(winner_index)
    return selected_parents


# Adjusts the parameters (population size, number of generations, crossover and mutation operations per generatiosn, parents kept
def set_parameters_GA(population_coef = 3, rounds_coef = 0.6, parents_coef = 20, 
                      crossover_coef = 20, mutation_order_coef = 1, mutation_path_coef = 20): # 5
    global MAX_JOBS, NR_PATH_PLANNED, POPULATION_SIZE, INIT_JOBS, ROUNDS

    MAX_JOBS = min(8, cpu_count() - 4) if cpu_count() > 6 else 1
    POPULATION_SIZE = min(int(NR_PATH_PLANNED * population_coef), 12) # 55

    INIT_JOBS = MAX_JOBS
    ROUNDS = min(int(POPULATION_SIZE * rounds_coef), 7) # 25

    global PARENTS_KEPT, CROSSOVERS, CROSSOVERS_POINTS, CROSSOVER_JOBS
    PARENTS_KEPT = int((POPULATION_SIZE * parents_coef) / 100) # 25
    CROSSOVERS = int((POPULATION_SIZE * crossover_coef) / 100)
    CROSSOVERS_POINTS = NR_PATH_PLANNED // 7 + 1

    CROSSOVER_JOBS = MAX_JOBS

    global MUTATIONS_ORDER, MUTATIONS_PATH, MUTATION_ORDER_JOBS, MUTATION_PATH_JOBS, NEW_INDIVIDUALS, NEW_INDIV_JOBS   
    MUTATIONS_ORDER = int(min(1, (NR_PATH_PLANNED * POPULATION_SIZE * mutation_order_coef) / 100)) # 10

    MUTATION_ORDER_JOBS = MAX_JOBS
    MUTATIONS_PATH = int(min(1, int((POPULATION_SIZE * mutation_path_coef) / 100))) # 10

    MUTATION_PATH_JOBS = MAX_JOBS

    NEW_INDIVIDUALS = POPULATION_SIZE - PARENTS_KEPT - CROSSOVERS - MUTATIONS_ORDER - MUTATIONS_PATH

    NEW_INDIV_JOBS = MAX_JOBS



def genetic_routing(filename, logs_file, seed, max_minutes = None, event = None):
    global population, best_solution, POPULATION_SIZE, ROUNDS, PARENTS_KEPT, \
        CROSSOVERS, CROSSOVERS_POINTS, CROSSOVER_JOBS, \
        MUTATIONS_ORDER, MUTATION_ORDER_JOBS, MUTATIONS_PATH, MUTATION_PATH_JOBS, \
        NEW_INDIVIDUALS, NEW_INDIV_JOBS, LAYERS, ROWS, COLUMNS, SELECTION

    start_time = time.time()
    random.seed(seed)

    for i in range(ROUNDS+1):
        elapsed_time = time.time() - start_time
        if not event or not event.is_set() or (max_minutes and elapsed_time * 60 < max_minutes):
            # Update progress
            progress = ceil((i + 1) / ROUNDS * 100)
            with open("progress.txt", "w") as f:
                f.write(str(ROUNDS)+','+str(progress))

            print(f"Generation {i+1}")
            count = 0
            changed = False
            for j in range(POPULATION_SIZE):
                if population[j].total_cost == -1:
                    paths_info = population[j].paths
                    paths = [x.path for x in paths_info if x]
                    population[j].unplaced_routes_number = NR_PATH_PLANNED - len(paths)
                    n = population[j].unplaced_routes_number
                    cost = fitness_function(paths, n, 4)
                    population[j].total_cost = cost
                    print(j, cost)

                    if best_solution.total_cost == -1 or (cost != -1 and cost < best_solution.total_cost) or n < len(planned_routes) - len(best_solution.paths):
                        best_solution = copy.deepcopy(population[0])
                        count = 1
                        changed = True
                    elif cost == best_solution.total_cost:
                        count += 1

            # Check if 85% of the population has the same cost as the best individual
            if count >= 0.85 * POPULATION_SIZE and best_solution.unplaced_routes_number < 1:
                print(f"Stopping early at generation {i+1} because 85% of the population has the same minimum cost.")
                progress = 100
                with open("progress.txt", "w") as f:
                    f.write(str(i+1)+','+str(progress))
                break

            if changed:
                update_board_file(filename, event)
            
            # Last generation  won't be updated
            if i == ROUNDS:
                break

            next_generation = []
            # tournament selection 
            aux_seed = random.random()
            selected_parents = tournament_selection(aux_seed, population, tournament_size=3, num_parents=PARENTS_KEPT)
            for index in selected_parents:
                next_generation.append(copy.deepcopy(population[index]))

            # crossover + mutation
            result = parallel_genetic_operators(aux_seed, CROSSOVERS, MUTATIONS_PATH, MUTATIONS_ORDER, population, (NR_PATH_PLANNED, template_grid, planned_routes, pads_list, netcodes_list, LAYERS, ROWS, COLUMNS))
            next_generation.extend(result)

            # Add new individuals into the new generation
            aux_seed = random.random()
            new_individuals = parallel_population_initialize(NEW_INDIVIDUALS, NEW_INDIV_JOBS, aux_seed, None, None, template_grid, planned_routes, pads_list, netcodes_list, LAYERS, ROWS, COLUMNS)
            next_generation.extend(new_individuals)
            
            population = next_generation

            print(time.time() - start_time)
        else:
            break

    print("End time", time.time() - start_time)


# Save sollution in a new file
def update_board_file(filename, event):
    global OFFSET_X, OFFSET_Y, AXIS_MULT, planned_routes
    suffix = "_routed"
    if best_solution and (not event or not event.is_set()):
        best_paths = best_solution.paths
        for index in range(len(best_paths)):
            best_paths[index].update_simplified_path()

        segments = get_segments_for_board(best_paths, planned_routes, (OFFSET_Y, OFFSET_X), AXIS_MULT)
        
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}{suffix}{ext}"

        write_segments_to_EOF(filename, new_filename, segments)

    finished_file = "done.txt"
    with open(finished_file, 'w') as f:
        f.write("done")
    f.close()


# Initialize + update population
def run_genetic_algorithm(filename, event = None, **kwargs):
    if not event or not event.is_set():
        global template_grid, pads_list, planned_routes, netcodes_list, OFFSET_Y, OFFSET_X, AXIS_MULT, LAYERS
        template_grid, AXIS_MULT, OFFSET_Y, OFFSET_X, pads_list, planned_routes, netcodes_list, LAYERS = get_data_for_GA(filename, **kwargs)

        global ROWS, COLUMNS, NR_PATH_PLANNED, SEED
        SEED = 10
        ROWS, COLUMNS = len(template_grid[0]), len(template_grid[0][0])
        NR_PATH_PLANNED = len(netcodes_list)
        max_minutes = None

        set_parameters_GA()

        print("Populare ...")
        global population, POPULATION_SIZE, INIT_JOBS, NEW_INDIVIDUALS, PARENTS_KEPT, CROSSOVERS, MUTATIONS_ORDER, MUTATIONS_PATH

        random.seed(SEED)
        aux_seed = random.random()
        a = time.time()
        population = parallel_population_initialize(POPULATION_SIZE, INIT_JOBS, aux_seed, None, None, template_grid, planned_routes, pads_list, netcodes_list, LAYERS, ROWS, COLUMNS)
        print("Populare finalizata", time.time() - a)

        aux_seed = random.random()
        genetic_routing(filename, logs_file = "a.txt", seed=aux_seed, max_minutes = max_minutes, event = event)

        update_board_file(filename, event)

        print(f"\n\norder = {best_solution.order}, cost = {best_solution.total_cost}, paths = {best_solution.paths}")

        if event:
            event.set()

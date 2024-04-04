from A_star import multiple_routes_A_star, read_file_routes, mark_obst_in_grid
from utils import fitness_function
import numpy as np
import random
import copy

POPULATION_SIZE:    int
ROUNDS:             int
MUTATIONS:          int
CROSSOVERS:         int
PARENTS_KEPT:       int
ROWS = None
COLUMNS = None

best_solution = []
template_grid = None
routes = []

''' not urgent
# in a group some routes should be ignored (avoid cycles)
def ignore_routes():
    ...

# check if routes can be grouped to find new possible routes
def group_routes():
    ...

# from one group create new possible routes used as alternatives for the ones defined in file
def define_new_possible_routes():
    ...
'''

'''
def adjust_parameters():
    ...
'''
''' not urgent '''



# maybe i will insert in mutation / crossover a way to move a cell to adjent one in order to reduce some angles
class Individual:
        def __init__(self, order = [], grid = [], paths = []):
            self.order  = order    # holds order of routes from input
            self.grid   = grid    # grid might be optional
            self.paths  = paths    # [[path1], [path2], ...]
            self.unplaced_routes_number = 0 # from the necessary ones
            self.path_cost = 0      # fitness_value

        def __repr__(self):
            return repr((self.order, self.grid, self.paths, self.unplaced_routes_number, self.path_cost))
        
        @staticmethod
        def return_path_cost(self):
            return self.path_cost


def random_routing_order(routes_number: int):
    order_list = list(range(routes_number))
    random.shuffle(order_list)
    return order_list



def order_routes(start_end_conn, route_order:list):
    ordered_route_list = copy.deepcopy(start_end_conn)
    for i in range(len(route_order)):
        index = route_order[i]
        ordered_route_list[i], ordered_route_list[index] = ordered_route_list[index], ordered_route_list[i]
    return ordered_route_list



def generate_individual(routes_number = 0, ordered_index_routes = None): # maybe add option for a starting grid and paths
    global routes
    n = len(routes)     # how to ignore the routes that are defined but not used
    if ordered_index_routes:
        random_order_routes = order_routes
    else:
        random_order_routes = random_routing_order(n)

    # routes number will be used to either remove cycles or will be dropped
    global template_grid, ROWS, COLUMNS
    #for i in range(n):
    grid, possible_solution = multiple_routes_A_star(grid = template_grid, routes = random_order_routes, pins_sizes = 1, 
                                                         rows = ROWS, columns = COLUMNS)
    return random_order_routes, possible_solution, grid  


def initialize_population(population_size: int, routes_number):
    global template_grid, routes
    population = []
    for i in range(population_size):
        routing_order, path, grid = generate_individual(routes = routes,
                                                        routes_number = routes_number, rows = ROWS, columns = COLUMNS)
        value = Individual(grid = grid, routing_order = routing_order, path = path)

        population[i] = value
    return population


# 0 - nothing in common, n-1 - everything
def common_part_index(list1, list2):
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return i
    return len(list1)


def crossover_operator(child1: Individual, child2: Individual):
    size = len(child1.order)
    index = random.randint(0, size - 1)

    child1_order = child1.order
    child2_order = child2.order
    child1.order = child1_order[:index] + child2_order[index:]
    child2.order = child2_order[:index] + child1_order[index:]
    
    if len(child1.order) != len(set(child1.order)):
        return False

    global routes
    ordered_routes_child1 = order_routes(start_end_conn=routes, route_order=child1.order)
    ordered_routes_child2 = order_routes(start_end_conn=routes, route_order=child2.order)

    paths1, grid1 = generate_individual(ordered_index_routes = ordered_routes_child1)
    paths2, grid2 = generate_individual(ordered_index_routes = ordered_routes_child2)

    child1.grid = grid1
    child2.grid = grid2

    child1.paths = paths1
    child2.paths = paths2

    return True


def crossover(parent1: Individual, parent2: Individual):
    tries = 5
    while tries > 0:
        tries -= 1

        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
            
        crossover_success = crossover_operator(child1, child2)
        if crossover_success == True:
            return child1, child2

    return parent1, parent2


def mutation(parent: Individual):       # later can be modified to replace one segment from a circuit to another (3 or more points routing)
    order = copy.deepcopy(parent.order)
    size = len(order)
    index1 = random.randint(0, size-1)
    index2 = random.randint(0, size-1)

    while index1 == index2:
        index2 = random.randint(0, size-1)
    
    order[index1], order[index2] = order[index1], order[index2] # swap the order for routing

    # to add a starting matrix from 0 to index
    global template_grid, routes
    routing_order, path, grid = generate_individual(ordered_index_routes=order)
    child = Individual(grid = grid, routing_order = routing_order, path = path)
    
    return child



# Selection based on roulette wheel; returns list on indexes of individuals selected from a generation
def roulette_wheel_selection(population, selected_number):
    # Sum of all fitness values of inividuals
    sum_fitness = sum(individual.path_cost for individual in population)

    selected = []
    for _ in range(selected_number):
        # Choose a point on roulette wheel
        selection_point = random.uniform(0, sum_fitness)
        sum_selection = 0
        index = 0
        for individual in population:
            sum_selection += individual.path_cost
            if sum_selection >= selection_point:
                selected.append(index)
                break

    return selected



def main():
    global routes
    routes, colors = read_file_routes(file_name='pins.csv', draw = False)
    routes_number = len(routes)
    blocked_areas = None

    # from file determine x y coordinates for grid, if there are no such values, extract from min max
    offset_x = 0
    offset_y = 0

    global ROWS, COLUMNS, POPULATION_SIZE, ROUNDS
    ROUNDS = 50
    ROWS = 100
    COLUMNS = 100
    
    global template_grid, best_solution
    template_grid = np.ones((ROWS, COLUMNS), dtype=int)
    
    # mark with red cells that can be used (obstacles)
    template_grid = mark_obst_in_grid(blocked_cells = blocked_areas, grid=template_grid, value=0)

    #individual_routes_placed = [[[] for j in range(routes_number)] for i in range(POPULATION_SIZE)] - instead of individual class
    #individual_grid = [[[] for j in range(routes_number)] for i in range(POPULATION_SIZE)]
    
    population = initialize_population(population_size=POPULATION_SIZE, pin_to_pin_conn=routes)

    best_solution = []  # is influenced by number of routes unplaced (1) and routes totala length 
    for i in range(ROUNDS):
        for j in range(POPULATION_SIZE):
            population[j].path_cost = fitness_function(routes = population[j].paths, 
                                                       unplaced_routes_number = population[j].unplaced_routes_numbeer, 
                                                       unplaced_route_penalty=1.5)
        population.sort(reverse = False, key = Individual.return_path_cost)
        
        next_generation = []
        indexes = roulette_wheel_selection(population=population, selected_number=PARENTS_KEPT)
        for index in indexes:
            next_generation.append(copy.deepcopy(population[index]))



if __name__ == "main":
    main()
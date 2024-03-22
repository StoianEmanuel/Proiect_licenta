from A_star import multiple_routes_A_star, read_file_routes, mark_obst_in_grid
import numpy as np

POPULATION_SIZE:    int
ROUNDS:             int
MUTATIONS:          int
CROSSOVERS:         int
PARENTS_KEPT:       int

population = []

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


''' not urgent '''
def adjust_parameters():
    ...


def generate_individual():
    ...

def crossover():
    ...

def mutation():
    ...

def fitness_function():
    ...

def routing():
    ...


def main():
    routes, colors = read_file_routes(file_name='pins.csv', draw = False)
    blocked_areas = None

    # from file determine x y coordinates for grid, if there are no such values, extract from min max
    offset_x = 0
    offset_y = 0

    rows = 100
    columns = 100
    
    grid = np.ones((rows, columns), dtype=int)
    # mark with red cells that can be used (obstacles)
    grid = mark_obst_in_grid(blocked_cells = blocked_areas, grid=grid, value=0)

    population = np.zeros(shape=(POPULATION_SIZE, rows, columns), dtype=int)


if __name__ == "main":
    main()
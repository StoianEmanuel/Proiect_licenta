from A_star import multiple_routes_A_star


POPULATION_SIZE:    int
ROUNDS:             int
MUTATIONS:          int
CROSSOVERS:         int
PARENTS_KEPT:       int

routes = []


# in a group some routes should be ignored (avoid cycles)
def ignore_routes():
    ...

# check if routes can be grouped to find new possible routes
def group_routes():
    ...

# from one group create new possible routes used as alternatives for the ones defined in file
def define_new_possible_routes():
    ...

def adjust_parameters():
    ...

def crossover():
    ...

def mutation():
    ...

def generate_individual():
    ...

def fitness_function():
    ...

def routing():
    ...

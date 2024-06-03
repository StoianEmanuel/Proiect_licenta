# stores functions related to A_star.py and GA_routing
from math import sqrt
from colored_repr_utils import COLORS
import csv
import os
import copy


'''#===================================''' 
def delete_file(file_name):
    try:
        if os.path.exists(file_name):
            os.remove(file_name)
    except Exception as e:  # due to permissions or used by another process
        print(e)
'''#===================================''' 



def check_element_in_list(targeted_element, list_of_elements):
    if not list_of_elements:
        return False
    
    return targeted_element in list_of_elements



def mark_path_in_array(array, path, value, overwrite = True):
    ''' value = value assigned to route'''
    aux = copy.deepcopy(array)
    try:
        for vertex in path:
            y, x = vertex
            if overwrite or aux[y][x] == 0:
                aux[y][x] = value
    except Exception as e:
        print("Error ", e, "while marking path in array")
    return aux



# class used for A* search, that stores 2 types of costs: so far and remaining
class Cell:
    '''Class used for A star search to determine the cost of paths based on heuristic\n
    Attributes:
        parent_y (int): Y coord of Cell's parent
        parent_x (int): X coord of Cell's parent (from where is accessed)
        f (float): Total cost; f = g + h
        h (float): Cost from starting cell
        g (float): Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest
    '''
    def __init__(self, parent = None, f = float('inf'), h = float('inf'), g = 0, 
                 direction = None, nr_bends: int = 0, nr_90_deg_bends: int = 0):
        if parent:
            self.parent = parent
        else:
            self.parent = (-1, -1)
        self.f = f  # Total cost (h + g)
        self.h = h  # Cost from start to cell
        self.g = g  # Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest
        self.direction = direction
        self.bends = nr_bends
        self.bends_90_deg = nr_90_deg_bends



# class used to define a path between two points that has width = n x 1;  
class Path:
    """Class used to define a path between two points with a specified width.\n
    Attributes:
        start_x      (int): The x-coordinate of the starting point.
        start_y      (int): The y-coordinate of the starting point.
        dest_x       (int): The x-coordinate of the destination point.
        dest_y       (int): The y-coordinate of the destination point.
        path        (list): The list of points representing the main path.
        width        (int): The width of the path.
        other_nodes (list): Additional points related to the main path.
    """
    def __init__(self, start: tuple[int, int], destination: tuple[int,  int],
                 path = None, width: int = 1, clearance: int = 1, other_nodes = None, mutated: bool = False):
        self.start          = start
        self.destination    = destination
        self.path           = path
        self.width          = width
        self.mutated        = mutated
        self.clearance      = clearance
        self.other_nodes    =  other_nodes # stores points related to main path



class Pad:
    '''
    Class used to define a pad for a part.
    Attributes:
        center_x       (int, int): (Y, X) coord of pad's center  
        pad_area           (list): (Y, X) coord occupied by pad on board; useful for irregular shapes or circles
    '''
    def __init__(self, center: tuple[int, int], original_center: tuple[int, int], 
                 pad_area = None, drill_area = None, pad_name: str = None, netcode = None, netname = None):
        self.center         = center
        self.original_center= original_center  # inainte de transformari (in nm)
        self.drill_area     = drill_area           # s-ar putea sa renunt la ea
        self.pad_area       = pad_area   # coord care realizeaza poligonul - lista de tupluri
        self.pad_name       = pad_name       # s-ar putea sa renunt la el
        self.netcode        = netcode    # folosit pentru a determina carui net ii sunt asociate
        self.netname        = netname    # posibil folosit pentru a crea folosi custom clearance si width

    def __str__(self) -> str:
        return f'Pad Name: {self.pad_name}\nCenter: ({self.center})'


# check if cell / move is valid
def is_unblocked(array, point: tuple[int, int], values: list):
    row, column = point
    return array[row][column] in values


# check if cell is inside the grid
def is_valid(point: tuple[int, int], array_shape: tuple[int, int]):
    row, column = point
    total_rows, total_columns = array_shape
    return 0 <= row < total_rows and 0 <= column < total_columns


# check if dest is reached
def is_destination(current_point: tuple[int, int], destination_point: tuple[int, int]):
    return current_point == destination_point

# poate voi insera si partea de bends
def route_length(route):        # route = [[,,,] - start,   ..., [,,,], ... ,      [,,,] - dest]
    distance = 0
    n = len(route)-1
    for index in range(n):
        p1_row, p1_column = route[index]
        p2_row, p2_column = route[index+1]
        distance += h_euclidian((p1_row, p1_column), (p2_row, p2_column))
    return distance



def fitness_function(routes, unplaced_routes_number: int, unplaced_route_penalty = 1.5):
    total_length = 0
    for route in routes:
        l = route_length(route)        
        total_length += l

    total_length = total_length * (unplaced_route_penalty ** unplaced_routes_number)
    return total_length



# function that save for each path only the points (x, y) that are start, destionation or represents a intersection between 2 lines
# forms an angle  
def simplify_path_list(paths_list):
    simplified_paths = []
    for path in paths_list:
        simplified_path = []      # most significant points - start, stop, "bend" points
        if path:
            length = len(path)
            simplified_path = [path[0]]
            for i in range(1, length-1):
                current_point = path[i]
                prev_point = path[i - 1]
                next_point = path[i + 1]

                direction_current_y  = next_point[0] - current_point[0]
                direction_current_x  = next_point[1] - current_point[1]
                direction_previous_y = current_point[0] - prev_point[0]
                direction_previous_x = current_point[1] - prev_point[1]

                if direction_current_y != direction_previous_y or direction_current_x != direction_previous_x:
                    simplified_path.append(current_point)

            simplified_path.append(path[-1])
            simplified_paths.append(simplified_path)
    return simplified_paths


# Returns tuple (dir_y_perp, dir_x_perp) so segment [(P.y, P.x), (P.y + dir_y, P.x + dir_x)] 
# is perpendicular to [(P.y, P.x), (P.y + dir_perp_y, P.x + dir_perp_x)]
def get_perpendicular_direction(direction_y: int, direction_x: int):
    if direction_y == 0:  # orizontal
        direction_perp_y = 1
        direction_perp_x = 0
    elif direction_x == 0:
        direction_perp_y = 0
        direction_perp_x = 1
    elif abs(direction_y) == abs(direction_x):
        direction_perp_y = -direction_x
        direction_perp_x = direction_y
    else:   # abs(dir_x) != abs(dir_y) --- (-1,1), (1,-1)
        direction_perp_y = direction_y
        direction_perp_x = -direction_x

    return direction_perp_y, direction_perp_x







'''#===================================''' 
# TODO
# create a list of int from a string list if possible
def string_list_to_int(string_list):
    int_list = None     # if int_list remains None, row will be avoided
    if len(string_list) == 4:    # check so there are all 4 coord for pin1 and pin2
        if all(s.isdigit() for s in string_list):
            int_list = [int(s) for s in string_list]
    return int_list

# TODO
# each line represents a connection between P1(x,y) and P2(x,y) + color as string
def read_file_routes(file_name = 'pins.csv', draw = False):
    routes = []
    colors = []
    print(os.getcwdb())
    try:
        with open(file_name, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for row in csv_reader:
                try:
                    pins_coord = string_list_to_int(string_list=row[0:4])
                    if pins_coord:
                        routes.append(pins_coord)
                        if draw == True:
                            colors.append(COLORS['green'])
                except:
                    print("invalid line")

    except FileNotFoundError:
        print("File does not exists")

    print(routes)
    # print(colors)
    return routes, colors
'''#===================================''' 


'''movement heuristics types'''
# 4 directions
def h_manhattan(point1: tuple[int, int], point2: tuple[int, int]):
    return abs(point1[1] - point2[1]) + abs(point1[0] - point2[0])

# any direction
def h_euclidian(point1: tuple[int, int], point2: tuple[int, int]):
    return sqrt((point1[1] - point2[1])**2 + (point1[0] - point2[0])**2)
''''''



'''#===================================''' 
# TODO
# return a rectangle area of cells
def generate_rectangle(row: int, col: int, length_y: int, length_x: int):
    area = [(i+row, j+col) for j in range(length_x) for i in range(length_y)]
    return area

# TODO
# allocated are = [(x,y), (x,y)] == areas used so it won't use them
def set_area_in_array(array, x_start: int, y_start: int, size_x: int, size_y: int, value: int, allocated_area: None):
    for row in range(size_x):
        for col in range(size_y):
            if (allocated_area and (x_start + row, y_start + col) not in allocated_area) or not allocated_area:
                array[x_start + row][y_start + col] = value
'''#===================================''' 


# TODO
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

def copy_common_part(x1: Individual, x2: Individual):
    ...
'''
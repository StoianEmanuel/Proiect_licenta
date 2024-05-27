# stores functions related to A_star.py and GA_routing
from math import sqrt
from colored_repr_utils import COLORS
import csv
import os



def delete_file(file_name):
    try:
        if os.path.exists(file_name):
            os.remove(file_name)
    except Exception as e:  # due to permissions or used by another process
        print(e)



def check_element_in_list(target_element, list_of_elements):
    if not list_of_elements:
        return False
    
    for element in list_of_elements:
        if target_element == element:
            return True
        
    return False



# class used for A* search, that stores 2 types of costs: so far and remaining
class Cell:
    '''
    Class used for A star search to determine the cost of paths based on heuristic
    Attributes:
        parent_x (int): X coord of Cell's parent (from where is accessed)
        parent_y (int): Y coord of Cell's parent
        f (float): Total cost; f = g + h
        h (float): Cost from starting cell
        g (float): Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest
    '''
    def __init__(self, x = 0, y = 0, f = float('inf'), h = float('inf'), g = 0):
        self.parent_x = x
        self.parent_y = y
        self.f = f  # Total cost (h + g)
        self.h = h  # Cost from start to cell
        self.g = g  # Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest



# class used to define a path between two points that has width = n x 1;  
class Path:
    """
    Class used to define a path between two points with a specified width.
    Attributes:
        start_x      (int): The x-coordinate of the starting point.
        start_y      (int): The y-coordinate of the starting point.
        dest_x       (int): The x-coordinate of the destination point.
        dest_y       (int): The y-coordinate of the destination point.
        path        (list): The list of points representing the main path.
        width        (int): The width of the path.
        other_nodes (list): Additional points related to the main path.
    """
    def __init__(self, start_x: int, start_y: int, dest_x: int, dest_y: int, 
                 path = None, width: int = 1, clearance: int = 1, other_nodes = None):
        self.start_x = start_x
        self.start_y = start_y
        self.dest_x  = dest_x
        self.dest_y  = dest_y
        self.path    = path
        self.width   = width
        self.clearance = clearance
        self.other_nodes =  other_nodes # stores points related to main path



class Pad:
    '''
    Class used to define a pad for a part.
    Attributes:
        center_x       (int): X coord of pad's center
        center_y       (int): Y coord of pad's center
        height         (int): pad's height
        width          (int): pad's width  
        angle          (int): pad's orientation (0, 90, 180, 270)
        occupied_area (list): (X, Y) coord occupied by pad on board; useful for irregular shapes or circles
    '''
    def __init__(self, center_x: int = 0, center_y: int = 0, original_center_x: int = 0, original_center_y: int = 0, 
                 length: int = 0, width: int = 0, angle: int = 0, 
                 occupied_area = None, pad_name: str = None, part_name: str = None):
        self.center_x      = center_x
        self.center_y      = center_y
        self.org_center_x  = original_center_x  # inainte de transformari (in nm)
        self.org_center_y  = original_center_y  
        self.length        = length
        self.width         = width
        self.angle         = angle          # s-ar putea sa renunt la ea
        self.occupied_area = occupied_area  # coord care realizeaza poligonul - lista de tupluri
        self.pad_name      = pad_name       
        self.part_name     = part_name      # s-ar putea sa renunt la ea

    def __str__(self) -> str:
        return f'Pad Name: {self.pad_name}\nPart Name: {self.part_name}\nCenter: ({self.center_x}, {self.center_y})\nHeight: {self.length}\nWidth: {self.width}\nAngle: {self.angle}'


# check if cell / move is valid
def is_unblocked(array, row: int, col: int, values: list):
    # value used for different paths; 0 - blocked | 1 - unblocked | 2, 3, ... - paths 
    return array[row][col] in values



# check if cell is inside the grid
def is_valid(row: int, col: int, rows: int, columns: int):
    '''
    Return True if row in [0, rows-1] and col in [0, columns-1]; else False
    '''
    return row >= 0 and row < rows and col >= 0 and col < columns



# check if dest is reached
def is_destination(row: int, col: int, dest_row: int, dest_col: int):
    return row == dest_row and col == dest_col   



def route_length(route):        # route = [[,,,] - start,   ..., [,,,], ... ,      [,,,] - dest]
    distance = 0
    n = len(route)-1
    for index in range(n):
        p1_row, p1_col = route[index]
        p2_row, p2_col = route[index+1]
        distance += h_euclidian(st_row = p1_row, st_col = p1_col, 
                                dest_row = p2_row, dest_col = p2_col)
    return distance



def fitness_function(routes, unplaced_routes_number: int, unplaced_route_penalty = 1.5):
    total_length = 0
    for route in routes:
        l = route_length(route)        
        total_length += l

    total_length = total_length * (unplaced_route_penalty ** unplaced_routes_number)
    # add cost for number of vias used
    return total_length



# function that save for each path only the points (x, y) that are start, destionation or represents a intersection between 2 lines
# forms an angle  
def simplify_path_list(paths_list):
    simplified_paths = []
    
    for path in paths_list:
        ms_points = []      # most significant points - start, stop, "bend" points
        if path:
            length = len(path)
            ms_points.append(path[0])
            for i in range(1, length-1):
                curr_point = path[i]
                prev_point = path[i - 1]
                next_point = path[i + 1]

                curr_x_direction = next_point[0] - curr_point[0]
                curr_y_direction = next_point[1] - curr_point[1]
                prev_x_direction = curr_point[0] - prev_point[0]
                prev_y_direction = curr_point[1] - prev_point[1]

                if curr_x_direction != prev_x_direction or curr_y_direction != prev_y_direction:
                    ms_points.append(curr_point)

            ms_points.append(path[length-1])
            simplified_paths.append(ms_points)

    return simplified_paths


# Returns tuple (dir_x_perp, dir_y_perp) so segment [(P.x, P.y), (P.x + dir_x, P.y + dir_y)] 
# is perpendicular to [(P.x, P.y), (P.x + dir_perp_x, P.y + dir_perp_y)]
def get_perpendicular_direction(dir_x: int, dir_y: int):
    if dir_x == 0:  # orizontal
        dir_x_perp = 1
        dir_y_perp = 0
    elif dir_y == 0:
        dir_x_perp = 0
        dir_y_perp = 1
    elif abs(dir_x) == abs(dir_y):
        dir_x_perp = -dir_y
        dir_y_perp = dir_x
    else:   # abs(dir_x) != abs(dir_y) --- (-1,1), (1,-1)
        dir_x_perp = dir_x
        dir_y_perp = -dir_y

    return dir_x_perp, dir_y_perp


# create a list of int from a string list if possible
def string_list_to_int(string_list):
    int_list = None     # if int_list remains None, row will be avoided
    if len(string_list) == 4:    # check so there are all 4 coord for pin1 and pin2
        if all(s.isdigit() for s in string_list):
            int_list = [int(s) for s in string_list]
    return int_list


''''''
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
''''''


''''''
def read_file(filename: str):
    pass
''''''

'''movement heuristics types'''
# 4 directions
def h_manhattan(st_row: int, st_col: int, dest_row: int, dest_col: int):
    return abs(st_row - dest_row) + abs(st_col - dest_col)


# any direction
def h_euclidian(st_row: int, st_col: int, dest_row: int, dest_col: int):
    return sqrt((st_row - dest_row)**2 + (st_col - dest_col)**2)


# 8 directions
def h_diagonal(st_row: int, st_col: int, dest_row: int, dest_col: int):
    dx = abs(st_row - dest_row)
    dy = abs(st_col - dest_col)
    D  = 1  # node length
    D2 = 1.41421 #sqrt(2) - diagonal distance between nodes
    return D * (dx + dy) + (D2 - 2*D) * min(dx, dy)
''''''


# return a rectangle area of cells
def generate_rectangle(row: int, col: int, length_x: int, length_y):
    area = [(i+row, j+col) for j in range(length_y) for i in range(length_x)]
    return area


# allocated are = [(x,y), (x,y)] == areas used so it won't use them
def set_area_in_array(array, x_start: int, y_start: int, size_x: int, size_y: int, value: int, allocated_area: None):
    for row in range(size_x):
        for col in range(size_y):
            if (allocated_area and (x_start + row, y_start + col) not in allocated_area) or not allocated_area:
                array[x_start + row][y_start + col] = value


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


def mark_pins_in_grid(grid, st_row: int, st_col: int, 
                      dest_row: int, dest_col: int,
                      low_limit, pins_sizes = 1, value = -1, allocated_area = None): 
    # value = 0 - blocked, 1 - unblocked
    set_area_in_array(array = grid, x_start = st_row + low_limit, y_start = st_col + low_limit,
                        size_x = pins_sizes, size_y = pins_sizes, value = value, allocated_area = allocated_area)
    set_area_in_array(array = grid, x_start = dest_row + low_limit, y_start = dest_col + low_limit,
                        size_x = pins_sizes, size_y = pins_sizes, value = value, allocated_area = allocated_area)    

'''
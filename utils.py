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



# class used for A* search, that stores 2 types of costs: so far and remaining
class Cell:
    def __init__(self, x = 0, y = 0, f = float('inf'), h = float('inf'), g = 0):
        self.parent_x = x
        self.parent_y = y
        self.f = float('inf')   # total cost (h + g)
        self.h = float('inf')   # Cost from start to cell
        self.g = 0  # Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest



# check if cell / move is valid
def is_unblocked(grid, row: int, col: int, value = 0):
    # value used for different paths; 0 - blocked | 1 - unblocked | 2, 3, ... - paths 
    return grid[row][col] == 0



# check if cell is inside the grid
def is_valid(row: int, col: int, rows: int, columns: int):
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
def simplify_path_list(paths):
    simplified_paths = []
    for path in paths:
        ms_points = []      # most significant points - start, stop, "bend" points
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



def set_area_in_array(array, x_start: int, y_start: int, size_x: int, size_y: int, value: int):
    for row in range(size_x):
        for col in range(size_y):
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
'''
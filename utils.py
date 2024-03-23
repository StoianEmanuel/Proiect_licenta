# stores functions related to A_star.py and GA_routing
from math import sqrt
from colored_repr_utils import COLORS
import csv
import os


# class used to store x and y coord for a certain point
class Parent:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y


# class used for A* search, that stores 2 types of costs: so far and remaining
class Cell:
    def __init__(self):
        self.parent = Parent()
        self.f = float('inf')   # total cost (h + g)
        self.h = float('inf')   # Cost from start to cell
        self.g = 0  # Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest



# check if cell / move is valid
def is_unblocked(grid, row: int, col: int):
    # value used for different paths; 0 - blocked | 1 - unblocked | 2, 3, ... - paths 
    return grid[row][col] == 1



# check if cell is inside the grid
def is_valid(row: int, col: int, rows: int, columns: int):
    return row >= 0 and row < rows and col >= 0 and col < columns



# check if dest is reached
def is_destination(row: int, col: int, dest):
    return row == dest.x and col == dest.y   



def route_length(route):        # route = [[,,,] - start,   ..., [,,,], ... ,      [,,,] - dest]
    distance = 0
    for segment in route:
        distance += h_euclidian(row = segment[0], col = segment[1], dest = Parent(x = segment[2], y = segment[3]))
    return distance


def fitness_function(routes, unplaced_routes):
    total_length = 0
    for route in routes:
        l = route_length(route)        
        total_length += l
    for i in range(unplaced_routes):
        total_length *= 1.5 # change the cost of unplaced routes to sth else
    # add cost for number of vias used
    return total_length


# create a list of int from a string list if possible
def string_list_to_int(string_list):
    int_list = None     # if int_list remains None, row will be avoided
    if len(string_list) == 4:    # check so there are all 4 coord for pin1 and pin2
        if all(s.isdigit() for s in string_list):
            int_list = [int(s) for s in string_list]
    return int_list


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


'''movement heuristics types'''
# 4 directions
def h_manhattan(row: int, col: int, dest: Parent):
    return abs(row - dest.x) + abs(col - dest.y)


# any direction
def h_euclidian(row: int, col: int, dest: Parent):
    return sqrt((row - dest.x)**2 + (col - dest.y)**2)


# 8 directions
def h_diagonal(row: int, col: int, dest: Parent):
    dx = abs(row - dest.x)
    dy = abs(col - dest.y)
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
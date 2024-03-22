# stores functions related to A_star.py and GA_routing
from math import sqrt
from colored_repr_utils import COLORS
import csv
import os

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
def h_manhattan(row: int, col: int, dest):
    return abs(row - dest.x) + abs(col - dest.y)

# any direction
def h_euclidian(row: int, col: int, dest):
    return sqrt((row - dest.x)**2 + (col - dest.y)**2)

# 8 directions
def h_diagonal(row: int, col: int, dest):
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
import numpy as np
from math import sqrt
import heapq
import matplotlib.pyplot as plt
from utils import h_diagonal, h_euclidian, h_manhattan, generate_rectangle
import random


class Parent:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y


class Cell:
    def __init__(self):
        self.parent = Parent()
        self.f = float('inf')   # total cost (h + g)
        self.h = float('inf')   # Cost from start to cell
        self.g = 0  # Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest


# Define size of grid
# mothod to adjust size to what is needed
ROWS = 100
COLS = 100

# COLORS used for different paths, in final form all routes same color
COLORS = {  'white' : [1,1,1],          'black' : [0.0,0.0,0.0],    'red'   : [1,0.0,0.0],  'green' : [0.0,1,0.0],
            'orange': [1,1,0.0],        'blue'  : [0.0,0.0,1],      'yellow': [0.0,0.8,1],  'purple': [1,0.0,1],
            'pink'  : [0.8,0.0,0.0],    'aqua'  : [0.0,1,1]
}


# choose a random color while avoiding other colors
def random_color(colors_ignored = None):
    available_colors = [color for color in COLORS.keys() if color not in colors_ignored]
    if available_colors:
        return random.choice(available_colors)
    else:
        return COLORS["black"]


# check if cell / move is valid
def is_unblocked(grid, row: int, col: int):
    # value used for different paths; 0 - blocked | 1 - unblocked | 2, 3, ... - paths 
    return grid[row][col] == 1


# check if cell is inside the grid
def is_valid(row: int, col: int, rows = ROWS, columns = COLS):
    return row >= 0 and row < rows and col >= 0 and col < columns


# check if dest is reached
def is_destination(row: int, col: int, dest):
    return row == dest.x and col == dest.y   


# insert non usable cells in grid
def add_obst_grid(blocked_cells, grid, value : int = 0): 
    # value = 1, 2, ... for routes, 0 for other types of obstacle
    grid_copy = grid.copy()
    if blocked_cells is not None:
        for cell in blocked_cells:
            i, j = cell
            grid_copy[i][j] = value
    return grid_copy


# Return the path from source to destination
def get_path(cell_details, dest, path_index = 1):   # 2024-03-13 16:00:31
    message = f"\nPath {path_index}. is:"
    print(message)
    path = []
    row, col = dest.x, dest.y

    # Trace path from dest to start using parent cells
    while not (cell_details[row][col].parent.x == row and cell_details[row][col].parent.y == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent.x
        temp_col = cell_details[row][col].parent.y
        row = temp_row
        col = temp_col
    
    path.append((row, col)) # add start node to path
    path.reverse()

    for i in path:
        print(" ->", i, end="")
        
    return path


# Draw the grid and update color_matrix with the latest path
def draw_grid(color_matrix, path, color = COLORS["yellow"]):    # 2024-03-13 16:00:53
    if path != None:
        for i in path: # assign color to path
            x = i[0]
            y = i[1]
            color_matrix[x][y] = color 
        
        x, y = path[0]
        color_matrix[x][y] = COLORS["aqua"]    # color assignement for pins
        x, y = path[-1]
        color_matrix[x][y] = COLORS["aqua"]

    arr = np.array(color_matrix, dtype=float)

    plt.imshow(arr, origin='upper', extent=[0.0, 1, 0.0, 1])
    plt.axis("off")
    plt.show()


def a_star_search(grid, start, dest, closed_list, cell_details):
    if not is_valid(start.x, start.y) or not is_valid(dest.x, dest.y):
        print("\nStart | Dest invalid")
        return False

    if not is_unblocked(grid, start.x, start.y) or not is_unblocked(grid, dest.x, dest.y):
        print("\nStart | Dest blocked")
        return False

    # initialize start of the list
    i = start.x
    j = start.y
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent.x = i
    cell_details[i][j].parent.y = j

    open_list = []  # cells to be visited
    heapq.heappush(open_list, (0.0, i, j))

    found_dest = False
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while len(open_list) > 0:
        p = heapq.heappop(open_list)
        
        # Mark cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        # for each direction check the succesor
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                if is_destination(new_i, new_j, dest):
                    cell_details[new_i][new_j].parent.x = i
                    cell_details[new_i][new_i].parent.y = j
                    print("\n\nDestination cell found")
                    found_dest = True
                    return True
                
                else:
                    g_new = cell_details[i][j].g + 1
                    h_new = h_euclidian(new_i, new_j, dest)
                    f_new = g_new + h_new

                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].parent.x = i
                        cell_details[new_i][new_j].parent.y = j

    if not found_dest:
        print("\nDestination not reached")
        return False 


# return an array filled with background color and colors assigned to pins 
def get_RGB_matrix(nodes, colors_list, background = COLORS['white'], rows = ROWS, columns = COLS):
    matrix = [[background for _ in range(columns)] for _ in range(rows)] # used to assign colors for routes

    for i in range(len(colors_list)):
        x = np.array(nodes).shape
        if x != (4,): #  at least 2 routes
            pin1_x, pin1_y, pin2_x, pin2_y = nodes[i] # nodes coords
        else:
            pin1_x = nodes[0];  pin1_y = nodes[1]
            pin2_x = nodes[2];  pin2_y = nodes[3]

        matrix[pin1_x][pin1_y] = colors_list[i]
        matrix[pin2_x][pin2_y] = colors_list[i]
    
    return matrix


# reset cells from grid if not part of solution (path) to a default value
def reset_cells_in_array(array, path, reset_value):
    if path:
        for i in range(len(array)):
            for j in range(len(array[0])):
                if (i, j) not in path:
                    array[i][j] = reset_value
    else:
        arr = np.full(np.array(array).shape, reset_value)
        array = arr


# mark path in array - used for Lee
def mark_path_in_array(array, path, value):
    if path:
        for x, y in path:
            array[x][y] = value


# run multiple A* routers for different sets of points
# function for routing
def multiple_routes_A_star(grid, routes, colors_list, color_matrix, rows = ROWS, columns = COLS, draw = False):
    closed_list = [[False for _ in range(columns)] for _ in range(rows)] # visited cells
    cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)] # status of every cell in the grid

    route_index = 0

    x = np.array(routes).shape
    if x != (4,): # at least 2 routes
        for cell in routes:
            route_index += 1
            start = Parent(cell[0], cell[1])    # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
            dest  = Parent(cell[2], cell[3])

            result = a_star_search(grid = grid, start = start, dest= dest, 
                                closed_list=closed_list, cell_details=cell_details)
            if result:
                path = get_path(cell_details = cell_details, dest=dest, path_index=route_index)

                if draw == True:
                    draw_grid(color_matrix=color_matrix, path=path, color=colors_list[route_index-1])

                mark_path_in_array(array = grid, path = path, value = route_index)

                # mark grid cells - for lee algorithm
                reset_cells_in_array(array=closed_list,  path=path, reset_value=False)
                cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)]
            else:
                print("\tNo change in drawing. Route can't be placed\n")
                reset_cells_in_array(array=closed_list, path=None, reset_value=False)
                cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)]
    
    else: # one single route
        route_index += 1
        start = Parent(routes[0], routes[1])    # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
        dest  = Parent(routes[2], routes[3])

        result = a_star_search(grid = grid, start = start, dest= dest, 
                            closed_list=closed_list, cell_details=cell_details)
        if result:
            path = get_path(cell_details = cell_details, dest=dest, path_index=route_index)
            if draw == True:
                draw_grid(color_matrix=color_matrix, path=path, color=colors_list[route_index-1])
        else:
            print("\tNo change in drawing. Route can't be placed\n")


# To add a way for a 3 or more pins connected to a wire: min spanning tree or A_star | Lee with propagation to wire | pins 

# create a list of int from a string list if possible
def string_list_to_int(string_list):
    int_list = None     # if int_list remains None, row will be avoided
    if len(string_list) == 4:    # check so there are all 4 coord for pin1 and pin2
        if all(s.isdigit() for s in string_list):
            int_list = [int(s) for s in string_list]
    return int_list


# each line represents a connection between P1(x,y) and P2(x,y) + color as string
import csv
def read_file_routes(file_name = 'pins.csv', draw = False):
    routes = []
    colors = []
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

    print(routes)
    print(colors)
    return routes, colors


def solution(routes, rows = ROWS, columns = COLS, blocked_areas = None, colors = None, draw = False):
    grid = np.ones((rows, columns), dtype=int)
    
    #colors = [COLORS['aqua'], COLORS['aqua'], COLORS['pink']]
    #routes = [[50, 0, 0, 0], [10, 20, 30, 30], [5, 10, 45, 45]] # [[x1, y1, x2, y2], ... ]

    if draw == True:
        color_matrix = get_RGB_matrix(nodes=routes, colors_list=colors, background=COLORS['black'])
        color_matrix = add_obst_grid(blocked_cells=blocked_areas, grid=color_matrix, value=COLORS['red'])
        draw_grid(color_matrix=color_matrix, path=None)
    
    # mark with red cells that can be used (obstacles)
    grid = add_obst_grid(blocked_cells=blocked_areas, grid=grid, value=0)

    multiple_routes_A_star(grid=grid, routes=routes, colors_list=colors, color_matrix=color_matrix, draw=draw)


if __name__ == "__main__":
    rectangle = generate_rectangle(row = 10, col = 0, length_x = 1, length_y = 2)
    blocked  = [(0,2), (0,3), (4,0), (4,5), (4,8), (10,10), (1,1), (2,0), (2,1), (2,2), (2,3), (10,1), (2,4), (1,4), (1,5)]
    blocked = list(set(blocked + rectangle)) # remove duplicates
    
    routes, colors = read_file_routes(file_name='pins.csv', draw = True)
    solution(routes = routes, colors = colors, blocked_areas = blocked, draw = True)

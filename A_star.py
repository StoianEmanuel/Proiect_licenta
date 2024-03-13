import numpy as np
from math import sqrt
import heapq
import matplotlib.pyplot as plt
from utils import h_diagonal, h_euclidian, h_manhattan, generate_rectangle
import random
import pandas as pd


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
ROWS = 100
COLS = 100

# COLORS used for different paths, in final form all routes same color
COLORS = {  'white' : [1,1,1],   'black' : [0,0,0],   'red'   : [1,0,0],   'green' : [0,1,0],
            'orange': [1,1,0],   'blue'  : [0,0,1],   'yellow': [0,1,1],   'purple': [1,0,1],
            'pink'  : [0.8,0,0],   'cyan'  : [0,0.8,0.8]
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
def is_valid(row: int, col: int):
    return row >= 0 and row < ROWS and col >= 0 and col < COLS


# check if dest is reached
def is_destination(row: int, col: int, dest):
    return row == dest.x and col == dest.y   


# insert non usable cells in grid
def add_obst_grid(blocked_cells, grid, value : int = 0): 
    # value = 1, 2, ... for routes, 0 for other types of obstacle
    for cell in blocked_cells:
        i, j = cell
        grid[i][j] = value



# Return the path from source to destination
def get_path(cell_details, dest, path_index = 1):   # 2024-03-13 16:00:31
    path = []
    message = f"Path {path_index}. is:\n"
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
        print("->", i, end="")
        
    return path


# Draw the grid and update color_matrix with the latest path
def draw_grid(color_matrix, path, color = COLORS["yellow"]):    # 2024-03-13 16:00:53
    for i in path: # assign color to path
        x = i[0]
        y = i[1]
        color_matrix[x][y] = color 

    color_matrix[path[0][0]][path[0][0]] = COLORS["green"]    # color assignement for pins
    color_matrix[path[len(path)-1][0]][path[(len(path)-1)][1]] = COLORS["green"]

    plt.imshow(color_matrix)
    plt.axis("off")
    plt.show()


# Trace the path from source to destination
def trace_path(cell_details, dest, color_matrix, color = COLORS["yellow"], value = 1):
    message = f"Path {value}. is"
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

    path.append((row, col))
    path.reverse()

    for i in path:
        print("->", i, end="")
    
    for i in path: # assign color to path
        x = i[0]
        y = i[1]
        color_matrix[x][y] = color 


    color_matrix[row][col] = COLORS["green"]    # color assignement for pins
    color_matrix[path[len(path)-1][0]][path[(len(path)-1)][1]] = COLORS["green"]

    '''rgb_matrix = np.zeros((ROWS, COLS, 3), dtype=float)
    for i in range(ROWS):
        for j in range(COLS):
            if color_matrix[i][j] == "white":
                rgb_matrix[i][j] = [0,0,0]
            elif color_matrix[i][j] == "red":
                rgb_matrix[i][j] = [1,0,0]
            elif color_matrix[i][j] == "yellow":
                rgb_matrix[i][j] = [1,1,0]
            elif color_matrix[i][j] == "green":
                rgb_matrix[i][j] = [0,1,0]'''

    plt.imshow(color_matrix)
    plt.axis("off")
    plt.show()

    return path # may not need it anymore


def a_star_search(grid, start, dest, closed_list, cell_details):
    if not is_valid(start.x, start.y) or not is_valid(dest.x, dest.y):
        print("Start | Dest invalid")
        return False

    if not is_unblocked(grid, start.x, start.y) or not is_unblocked(grid, dest.x, dest.y):
        print("Start | Dest blocked")
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
                    print("Destination cell found")
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
        print("Destination not reached")
        return False 


# return an array filled with background color and colors assigned to pins 
def assing_color_to_matrix(nodes, colors, background = COLORS['white']):
    matrix = [[background for _ in range(COLS)] for _ in range(ROWS)] # used to assign colors for routes
    for i in range(colors):
        pin1_x, pin1_y, pin2_x, pin2_y = nodes[i] # nodes coords
        matrix[pin1_x][pin1_y] = colors[i]
        matrix[pin2_x][pin2_y] = colors[i]
    return matrix


# reset cells from grid if not part of solution (path) to a default value
def reset_cells_not_used(array, path, default_value):
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] not in path:
                array[i][j] = default_value



# run multiple A* routers for different sets of points
# need to assign a color, a method to avoid previous path and be different from std obstacles
# function for routing
def multiple_routes_A_star(grid, nodes, colors, blocked):
    closed_list = [[False for _ in range(COLS)] for _ in range(ROWS)] # visited cells
    cell_details = [[Cell() for _ in range(COLS)] for _ in range(ROWS)] # status of every cell in the grid
    
    '''color_matrix = [[COLORS['white'] for _ in range(COLS)] for _ in range(ROWS)] # used to assign colors for routes'''

    color_matrix = assing_color_to_matrix(nodes=None, colors=[COLORS['green']], background=COLORS['white'])
    color_matrix = add_obst_grid(blocked_cells=blocked, grid=color_matrix, value=COLORS['red']) # mark with red cells that can be used (obstacles)
    ''' for i in blocked:
        x = i[0]
        y = i[1]
        color_matrix[x][y] = COLORS["red"]  # mark with red cells that can be used (obstacles)'''
    
    route_index = 0
    grid_copy = np.copy(grid)   # modifications will apply to a copy of original grid

    for cell in nodes:
        route_index += 1
        start = Parent(cell[0], cell[1])    # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
        dest  = Parent(cell[2], cell[3])
        path  = None

        result = a_star_search(grid = grid_copy, start = start, dest= dest, closed_list=closed_list, cell_details=cell_details)
        if result:
            trace_path(cell_details=cell_details, dest=dest, color_matrix=color_matrix)
            path = () # return path
            # draw()  # redraw/draw with modification
        else:
            print("\tNo change in drawings | Route can't be placed\n")

        if path:
            closed_list  = reset_cells_not_used(array=closed_list,  path=path, default_value=False)
            cell_details = reset_cells_not_used(array=cell_details, path=path, default_value=Cell())
        # grid = trace_route_grid(grid = grid, path = path)
        # to add a return function for result of prev iteration


def main():
    grid = np.ones((ROWS, COLS), dtype=int)
    rectangle = generate_rectangle(row=10, col=0, length_x=1, length_y=2)
    blocked  = [(0,2), (0,3), (4,0), (4,5), (4,8), (10,10), (1,1), (2,0), (2,1), (2,2), (2,3), (10,1), (2,4), (1,4), (1,5)]
    blocked = list(set(blocked + rectangle)) # remove duplicates
    
    start = Parent(x = 50, y = 0)
    dest  = Parent(x = 0,  y = 0)
    #multiple_routes_A_star(grid=grid, nodes=[[start.x, start.y, dest.x, dest.y]], blocked=blocked)
    a_star_search(grid, start, dest, blocked) 



if __name__ == "__main__":
    main()

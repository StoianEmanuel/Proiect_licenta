import numpy as np
import heapq
from colored_repr_utils import draw_grid, get_RGB_matrix, COLORS
from utils import h_diagonal, h_euclidian, h_manhattan, generate_rectangle, read_file_routes


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


# Define size of grid
# method to adjust size to what is needed
ROWS = 100
COLS = 100


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
def mark_obst_in_grid(blocked_cells, grid, value : int = 0): 
    # value = 1, 2, ... for routes, 0 for other types of obstacle
    grid_copy = grid.copy()
    if blocked_cells is not None:
        for cell in blocked_cells:
            i, j = cell
            grid_copy[i][j] = value
    return grid_copy



# Return the path from source to destination
def get_path_A_star(cell_details, dest, path_index = 1):   # 2024-03-13 16:00:31
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



# find one route at a time using A star algorihm (modified Dijkstra)
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



# ----------------------------------- to be added
def modified_lee_routing(grid, start, dest):
    path = []
    grid_copy = grid.copy()

    return path



# run multiple A* routers for different sets of points
# function for routing
def multiple_routes_A_star(grid, routes, colors_list, color_matrix, rows = ROWS, columns = COLS, draw = False):
    unplaced_routes = 0

    x = np.array(routes).shape
    if x != (4,): # at least 2 routes
        closed_list = [[False for _ in range(columns)] for _ in range(rows)] # visited cells
        route_index = 0
        for cell in routes:
            cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)] # status of every cell in the grid
            route_index += 1
            start = Parent(cell[0], cell[1])    # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
            dest  = Parent(cell[2], cell[3])

            result = a_star_search(grid = grid, start = start, dest= dest, 
                                closed_list=closed_list, cell_details=cell_details)
            if result:
                path = get_path_A_star(cell_details = cell_details, dest=dest, path_index=route_index)

                if draw == True:
                    draw_grid(color_matrix=color_matrix, path=path, color=colors_list[route_index-1])

                mark_path_in_array(array = grid, path = path, value = route_index)

                # mark grid cells - for lee algorithm
                reset_cells_in_array(array=closed_list, path=path, reset_value=False)
                #cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)]
            else:
                print("\tNo change in drawing. Route can't be placed\n")
                unplaced_routes += 1
                reset_cells_in_array(array=closed_list, path=None, reset_value=False)
                #cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)]
    
    else: # one single route
        route_index = 1
        start = Parent(routes[0], routes[1])    # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
        dest  = Parent(routes[2], routes[3])

        result = a_star_search(grid = grid, start = start, dest= dest, 
                            closed_list=closed_list, cell_details=cell_details)
        if result:
            path = get_path_A_star(cell_details = cell_details, dest=dest, path_index=route_index)
            if draw == True:
                draw_grid(color_matrix=color_matrix, path=path, color=colors_list[route_index-1])
        else:
            unplaced_routes += 1
            print("\tNo change in drawing. Route can't be placed\n")

    return grid, unplaced_routes



# check if initial representation is needed and start placing routes
def solution(grid, routes, rows = ROWS, columns = COLS, blocked_areas = None, colors = None, draw = False):
    if draw == True:
        color_matrix = get_RGB_matrix(nodes=routes, colors_list=colors, background=COLORS['black'])
        color_matrix = mark_obst_in_grid(blocked_cells=blocked_areas, grid=color_matrix, value=COLORS['red'])
        draw_grid(color_matrix=color_matrix, path=None)

    multiple_routes_A_star(grid=grid, routes=routes, rows=rows, columns=columns,
                           colors_list=colors, color_matrix=color_matrix, draw=draw)



# for testing purposes
if __name__ == "__main__":
    rows = ROWS
    columns = COLS

    #colors = [COLORS['aqua'], COLORS['aqua'], COLORS['pink']]
    #routes = [[50, 0, 0, 0], [10, 20, 30, 30], [5, 10, 45, 45]] # [[x1, y1, x2, y2], ... ]

    rectangle = generate_rectangle(row = 10, col = 0, length_x = 1, length_y = 2)
    blocked  = [(0,2), (0,3), (4,0), (4,5), (4,8), (10,10), (1,1), (2,0), (2,1), (2,2), (2,3), (10,1), (2,4), (1,4), (1,5)]
    blocked = list(set(blocked + rectangle)) # remove duplicates

    routes, colors = read_file_routes(file_name='pins.csv', draw = True)

    grid = np.ones((rows, columns), dtype=int)

    # mark with red cells that can be used (obstacles)
    grid = mark_obst_in_grid(blocked_cells = blocked, grid=grid, value=0)
    solution(grid = grid, routes = routes, colors = colors, blocked_areas = blocked, draw = True)

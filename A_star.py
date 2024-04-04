import numpy as np
import heapq
import copy
from colored_repr_utils import draw_grid, get_RGB_matrix, COLORS
from utils import   Cell, Parent, is_destination, is_unblocked, is_valid, \
                    h_diagonal, h_euclidian, h_manhattan, \
                    generate_rectangle, read_file_routes, set_area_in_array


# Define size of grid
# method to adjust size to what is needed
ROWS = 70
COLS = 70


# insert non usable cells in grid
def mark_obst_in_grid(blocked_cells, grid, value : int = 0): 
    # value = 1, 2, ... for routes, 0 for other types of obstacle
    grid_copy = copy.deepcopy(grid)
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
def a_star_search(grid, start, dest, closed_list, cell_details, rows, columns):
    if not is_valid(start.x, start.y, rows, columns) or not is_valid(dest.x, dest.y, rows, columns):
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

            if is_valid(new_i, new_j, rows, columns) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                if is_destination(new_i, new_j, dest):
                    cell_details[new_i][new_j].parent.x = i
                    cell_details[new_i][new_i].parent.y = j
                    print("\n\nDestination cell found")
                    found_dest = True
                    return True
                
                else:
                    g_new = cell_details[i][j].g + 1
                    h_new = h_euclidian(row = new_i, col = new_j, dest = dest)
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
    grid_copy = copy.deepcopy(grid)

    return path



def mark_pins_in_grid(grid, start: Parent, dest: Parent, low_limit, pins_sizes = 1, value = 0): 
    # value = 0 - blocked, 1 - unblocked
    set_area_in_array(array = grid, x_start = start.x - low_limit, y_start = start.y - low_limit,
                        size_x = pins_sizes, size_y = pins_sizes, value = value)
    set_area_in_array(array = grid, x_start = dest.x - low_limit, y_start = dest.y - low_limit,
                        size_x = pins_sizes, size_y = pins_sizes, value = value)



# run multiple A* routers for different sets of points
# function for routing
def multiple_routes_A_star(grid, routes, colors_list = None, color_matrix = None, rows = ROWS, columns = COLS, pins_sizes = 1, draw = False):
    paths = []

    x = np.array(routes).shape
    if x != (4,): # at least 2 routes
        closed_list = [[False for _ in range(columns)] for _ in range(rows)] # visited cells
        

        high_limit = int(pins_sizes/2) + 1
        low_limit  = -int(pins_sizes/2)
        grid_copy = copy.deepcopy(grid)

        for route in routes:
            start = Parent(x = route[0], y = route[1])    # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
            dest  = Parent(x = route[2], y = route[3])
            mark_pins_in_grid(grid = grid_copy, low_limit = low_limit, pins_sizes = 1, start = start, dest = dest, value = 0)

        for route in routes:
            cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)] # status of every cell in the grid
            route_index += 1
            start = Parent(x = route[0], y = route[1])    # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
            dest  = Parent(x = route[2], y = route[3])

            mark_pins_in_grid(grid = grid_copy, low_limit = low_limit, pins_sizes = 1, start = start, dest = dest, value = 1)

            result = a_star_search(grid = grid_copy, start = start, dest = dest, rows = rows, columns = columns,
                                    closed_list = closed_list, cell_details = cell_details)
            
            if result:
                path = get_path_A_star(cell_details = cell_details, dest = dest, path_index = route_index)
                paths.append(path)

                block = []
                for i in range(low_limit, high_limit):     # x axis
                    for j in range(low_limit, high_limit): # y axis
                        if (start.x + i, start.y + j) not in path:
                            block.append((start.x + i, start.y + j))
                        if (dest.x + i, dest.y + j) not in path:
                            block.append((dest.x + i, dest.y + j))

                grid_copy = mark_obst_in_grid(blocked_cells=block, grid=grid_copy, value=0)
                
                if draw == True:    draw_grid(color_matrix = color_matrix, path = path, color = colors_list[route_index-1]) 

                mark_path_in_array(array = grid, path = path, value = route_index)

                # mark grid cells - for lee algorithm
                reset_cells_in_array(array = closed_list, path = path, reset_value = False)

            else:
                print("\tNo change in drawing. Route can't be placed\n")
                reset_cells_in_array(array = closed_list, path = None, reset_value = False)
                paths.append([])
    
    else: # one single route
        route_index = 1
        start = Parent(x = routes[0], y = routes[1])    # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
        dest  = Parent(x = routes[2], y = routes[3])

        result = a_star_search(grid = grid, start = start, dest = dest, closed_list = closed_list, cell_details = cell_details,
                               rows = rows, columns = columns)
        if result:
            path = get_path_A_star(cell_details = cell_details, dest = dest, path_index = route_index)
            paths.append(path)

            if draw == True:    draw_grid(color_matrix = color_matrix, path = path, color = colors_list[route_index-1])

        else:
            print("\tNo change in drawing. Route can't be placed\n")
            paths.append([])
    # uplaced_routes to be changed to sth to check if there are all nodes present in solution and if not, counts the remaining ones
    return grid, path



# check if initial representation is needed and start placing routes
def solution(grid, routes, pins_sizes = 1, rows = ROWS, columns = COLS, blocked_areas = None, colors = None, draw = False):
    if draw == True:
        color_matrix = get_RGB_matrix(nodes = routes, colors_list = colors, background = COLORS['black'], rows = rows, columns = columns)
        color_matrix = mark_obst_in_grid(blocked_cells = blocked_areas, grid = color_matrix, value = COLORS['red'])
        draw_grid(color_matrix = color_matrix, path=None)

    multiple_routes_A_star(grid = grid, routes = routes, rows = rows, columns = columns, pins_sizes = pins_sizes,
                           colors_list = colors, color_matrix = color_matrix, draw = draw)



# for testing purposes
if __name__ == "__main__":
    rows = ROWS
    columns = COLS
    pins_sizes = 3
    #colors = [COLORS['aqua'], COLORS['aqua'], COLORS['pink']]
    #routes = [[50, 0, 0, 0], [10, 20, 30, 30], [5, 10, 45, 45]] # [[x1, y1, x2, y2], ... ]

    rectangle = generate_rectangle(row = 10, col = 0, length_x = 1, length_y = 2)
    blocked  = [(4,0), (4,5), (4,8), (10,10), (10,1), (2,4), (1,4), (1,5)]
    blocked = list(set(blocked + rectangle)) # remove duplicates

    routes, colors = read_file_routes(file_name='pins.csv', draw = True)

    grid = np.ones((rows, columns), dtype=int)

    # mark with red cells that can be used (obstacles)
    grid = mark_obst_in_grid(blocked_cells = blocked, grid = grid, value = 0)
    solution(grid = grid, routes = routes, pins_sizes = pins_sizes, 
             colors = colors, blocked_areas = blocked, draw = True)

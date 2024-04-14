import numpy as np
import heapq
import copy
import random
from colored_repr_utils import draw_grid, get_RGB_matrix, COLORS
from utils import   Cell, is_destination, is_unblocked, is_valid,\
                    h_diagonal, h_euclidian, h_manhattan, \
                    generate_rectangle, read_file_routes, set_area_in_array


# Define size of grid
# method to adjust size to what is needed
ROWS = 55
COLS = 55


# insert non usable cells in grid
def mark_areas_in_grid(blocked_cells, grid, value : int = 0): 
    # value = 1, 2, ... for routes, 0 for other types of obstacle
    grid_copy = copy.deepcopy(grid)
    if blocked_cells is not None:
        for cell in blocked_cells:
            i, j = cell
            grid_copy[i][j] = value
    return grid_copy



# Return the path from source to destination
def get_path_A_star(cell_details, dest_row, dest_col):   # 2024-03-13 16:00:31
    path = []
    row = dest_row
    col = dest_col

    #file1 = open("output.txt", "a")
    #for i in range(0, len(cell_details)):
    #    for j in range(0, len(cell_details[0])):
    #        file1.write(f"{i}, {j}, {cell_details[i][j].parent_x}, {cell_details[i][j].parent_y}\n")

    # Trace path from dest to start using parent cells
    while not (cell_details[row][col].parent_x == row and cell_details[row][col].parent_y == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_x
        temp_col = cell_details[row][col].parent_y
        row = temp_row
        col = temp_col
    
    path.append((row, col)) # add start node to path
    path.reverse()
        
    return path



# Print path on a custom format
def print_path(path, path_index = 0):
    if path_index > 0:
        message = f"\nPath {path_index}. is:"
        print(message)
        for i in path:
            print(i, end="->")



# find one route at a time using A star algorihm (modified Dijkstra)
def a_star_search(grid, st_row, st_col, dest_row, dest_col, closed_list, cell_details, rows, columns, 
                  suppress_prints: True):
    if not is_valid(st_row, st_col, rows, columns) or not is_valid(dest_row, dest_col, rows, columns):
        if suppress_prints == False:
            print("\nStart | Dest invalid")
        return False

    if not is_unblocked(grid, st_row, st_col, value=0) or not is_unblocked(grid, dest_row, dest_col, value=0):
        if suppress_prints == False:
            print("\nStart | Dest blocked")
        return False

    # initialize start of the list
    i = st_row
    j = st_col
    
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_x = i
    cell_details[i][j].parent_y = j

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

            if is_valid(new_i, new_j, rows, columns) and is_unblocked(grid, new_i, new_j, value=0) and \
                not closed_list[new_i][new_j]:
                # form 3 x 3 square to check if the paths are not overlapping
                ok = False
                if dir[0] == 0 or dir[1] == 0: # paths can't be overlapeed
                    ok = True
                else:
                    min_i, max_i = min(new_i, i), max(new_i, i)
                    min_j, max_j = min(new_j, j), max(new_j, j)

                    if dir[0] != dir[1]:    # /; new_i = i + 1, new_j = j + 1 or new_i = i - 1 and new_j = j - 1
                        if is_valid(row = min_i, col = min_j, rows = rows, columns = columns) and \
                            is_valid(row = max_i, col = max_j, rows = rows, columns = columns) and \
                                grid[min_i][min_j] != grid[max_i][max_j] or grid[min_i][min_j] <= 0 \
                                    or grid[max_i][max_j] <= 0:
                                ok = True
                    else: # \; new_i = i - 1, new_j = j + 1 or new_i = i + 1 and new_j = j - 1
                        if is_valid(row = min_i, col = max_j, rows = rows, columns = columns) and \
                            is_valid(row = max_i, col = min_j, rows = rows, columns = columns) and \
                                grid[min_i][max_j] != grid[max_i][min_j] or grid[min_i][max_j] <= 0 \
                                    or grid[max_i][min_j] <= 0:
                                ok = True

                if ok == True:
                    if not is_destination(new_i, new_j, dest_row, dest_col):
                        g_new = cell_details[i][j].g + 1
                        h_new = h_euclidian(st_row = new_i, st_col = new_j, dest_row = dest_row, dest_col = dest_col)
                        f_new = g_new + h_new

                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].parent_x = i
                            cell_details[new_i][new_j].parent_y = j

                    else:
                        cell_details[new_i][new_j].parent_x = i
                        cell_details[new_i][new_j].parent_y = j
                        if suppress_prints == False:
                            print("\n\nDestination cell found")
                        found_dest = True
                        return True

    if found_dest == False and suppress_prints == False:
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




# ----------------------------------- to be added
def modified_lee_routing(grid, start, dest):
    path = []
    grid_copy = copy.deepcopy(grid)

    return path



def mark_pins_in_grid(grid, st_row: int, st_col: int, 
                      dest_row: int, dest_col: int,
                      low_limit, pins_sizes = 1, value = -1): 
    # value = 0 - blocked, 1 - unblocked
    set_area_in_array(array = grid, x_start = st_row + low_limit, y_start = st_col + low_limit,
                        size_x = pins_sizes, size_y = pins_sizes, value = value)
    set_area_in_array(array = grid, x_start = dest_row + low_limit, y_start = dest_col + low_limit,
                        size_x = pins_sizes, size_y = pins_sizes, value = value)



# run multiple A* routers for different sets of points
# function for routing
def multiple_routes_A_star(grid, routes, colors_list = None, color_matrix = None, rows = ROWS, columns = COLS, 
                           pins_sizes = 1, draw = False, suppress_prints = True):
    paths = []
    route_index = 0
    x = np.array(routes).shape
    grid_copy = copy.deepcopy(grid)
    if x != (4,): # at least 2 routes
        closed_list = [[False for _ in range(columns)] for _ in range(rows)] # visited cells

        high_limit = int(pins_sizes/2) + 1
        low_limit  = -int(pins_sizes/2)
        #print(np.array(routes).shape)

        for route in routes: # cells structure: [[x_s1,y_s1,x_d1,y_d1], [x_s2,y_s2,x_d2,y_d2], ...]
            st_row, st_col, dest_row, dest_col = route[0:4]
            mark_pins_in_grid(grid = grid_copy, low_limit = low_limit, pins_sizes = pins_sizes, 
                              st_row = st_row, st_col = st_col, dest_row = dest_row, dest_col = dest_col,
                              value = -1)

        for route in routes:    
            cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)] # status of every cell in the grid
            route_index += 1
            st_row, st_col, dest_row, dest_col = route[0:4]

            mark_pins_in_grid(grid = grid_copy, low_limit = low_limit, pins_sizes = pins_sizes, 
                              st_row = st_row, st_col = st_col, dest_row = dest_row, dest_col = dest_col, value = 0)
            

            result = a_star_search(grid = grid_copy, st_row = st_row, st_col = st_col, dest_row = dest_row, dest_col = dest_col,
                                    rows = rows, columns = columns,
                                    closed_list = closed_list, cell_details = cell_details, suppress_prints = suppress_prints)
            
            if result == True:
                path = get_path_A_star(cell_details = cell_details, dest_row = dest_row, dest_col = dest_col)
                if suppress_prints == False:
                    print_path(path = path, path_index = route_index)

                paths.append(path)
                print("\n", route_index, "\n")
                block = []

                for i in range(low_limit, high_limit):     # x axis
                    for j in range(low_limit, high_limit): # y axis
                        if (st_row + i, st_col + j) not in path:
                            block.append((st_row + i, st_col + j))
                        if (dest_row + i, dest_col + j) not in path:
                            block.append((dest_row + i, dest_col + j))

                grid_copy = mark_areas_in_grid(blocked_cells = block, grid = grid_copy, value = route_index)
                grid_copy = mark_areas_in_grid(blocked_cells = path, grid = grid_copy, value = route_index)
                
                if draw == True:    draw_grid(color_matrix = color_matrix, path = path, color = random.choice([COLORS['yellow'], COLORS['red'], COLORS['pink'], COLORS['orange']])) 

                #mark_path_in_array(array = grid_copy, path = path, value = route_index)
                
                # mark grid cells - for lee algorithm
                reset_cells_in_array(array = closed_list, path = path, reset_value = False)

            else:
                if suppress_prints == False:
                    print("\tNo change in drawing. Route can't be placed\n")
                reset_cells_in_array(array = closed_list, path = None, reset_value = False)
                paths.append([])
    
    else: # one single route
        route_index = 1
        st_row, st_col, dest_row, dest_col = route[0:4]

        result = a_star_search(grid = grid_copy, st_row = st_row, st_col = st_col, 
                               dest_row = dest_row, dest_col = dest_col, 
                               closed_list = closed_list, cell_details = cell_details,
                               rows = rows, columns = columns)
        if result:
            path = get_path_A_star(cell_details = cell_details, dest_row = dest_row, dest_col = dest_col, suppress_prints = suppress_prints)
            if suppress_prints == False:
                    print_path(path = path, path_index = 1)
            paths.append(path)

            if draw == True:    draw_grid(color_matrix = color_matrix, path = path, color = colors_list[route_index-1])

        else:
            if suppress_prints == False:
                print("\tNo change in drawing. Route can't be placed\n")
            paths.append([])
    # unplaced_routes to be changed to sth to check if there are all nodes present in solution and if not, counts the remaining ones
    return grid_copy, paths



# check if initial representation is needed and start placing routes
def solution(grid, routes, pins_sizes = 1, rows = ROWS, columns = COLS, blocked_areas = None, colors = None, draw = False):
    if draw == True:
        color_matrix = get_RGB_matrix(nodes = routes, colors_list = colors, background = COLORS['black'], rows = rows, columns = columns)
        color_matrix = mark_areas_in_grid(blocked_cells = blocked_areas, grid = color_matrix, value = COLORS['red'])
        draw_grid(color_matrix = color_matrix, path=None)

    grid_copy, paths = multiple_routes_A_star(grid = grid, routes = routes, rows = rows, columns = columns, pins_sizes = pins_sizes,
                           colors_list = colors, color_matrix = color_matrix, draw = draw, suppress_prints = False)
    



# for testing purposes
if __name__ == "__main__":
    rows = ROWS
    columns = COLS
    pins_sizes = 3
    #colors = [COLORS['aqua'], COLORS['aqua'], COLORS['pink']]
    #routes = [[50, 0, 0, 0], [10, 20, 30, 30], [5, 10, 45, 45]] # [[x1, y1, x2, y2], ... ]

    #rectangle = generate_rectangle(row = 10, col = 0, length_x = 1, length_y = 2)
    #blocked  = [(4,0), (4,4), (4,8), (10,10), (10,1), (2,4), (1,4), (1,5)]
    #blocked = list(set(blocked + rectangle)) # remove duplicates
    blocked = None

    routes, colors = read_file_routes(file_name='pins.csv', draw = True)

    grid = np.zeros((rows, columns), dtype=int)

    # mark with red cells that can be used (obstacles)
    grid = mark_areas_in_grid(blocked_cells = blocked, grid = grid, value = -1)
    solution(grid = grid, routes = routes, pins_sizes = pins_sizes, 
             colors = colors, blocked_areas = blocked, draw = True)

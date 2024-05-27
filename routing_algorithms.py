import numpy as np # --
import heapq
from collections import deque
import copy
import random
from colored_repr_utils import draw_grid, get_RGB_matrix, color_pads_in_RGB_matrix, COLORS
from utils import Cell, Path, Pad, is_destination, is_unblocked, is_valid,\
                  h_diagonal, h_euclidian, h_manhattan, \
                  generate_rectangle, read_file_routes, set_area_in_array, check_element_in_list, \
                  get_perpendicular_direction


# Define size of grid
ROWS = 55
COLS = 55


# insert non usable cells in grid
def set_values_in_array(blocked_cells, arr, value: 0): 
    # value = 1, 2, ... for routes, 0 for other types of obstacle
    arr_copy = copy.deepcopy(arr)
    if blocked_cells is not None:
        for cell in blocked_cells:
            i, j = cell
            arr_copy[i][j] = value
    return arr_copy


def get_offset(current_row: int, current_column: int, previous_row: int, previous_column: int):
    return current_row - previous_row, current_column - previous_column


def get_nodes(grid, row: int, column: int, width: int, dir_x: int, dir_y: int, values: int):
    '''
    Returns a list for cell neighbors according to width of the path
    grid    (array):
    row       (int):
    column    (int):
    width     (int):
    dir_x     (int):
    dir_y     (int):
    '''
    nodes = []
    side = (width - 1) // 2
    for i in range(1, side + 1):
        new_row_1 = row + i * dir_x
        new_col_1 = column + i * dir_y
      
        new_row_2 = row - i * dir_x
        new_col_2 = column - i * dir_y

        if new_row_1 > new_row_2 or (new_row_1 == new_row_2 and new_col_1 > new_col_2):
            nodes.append((new_row_1, new_col_1))
            nodes.insert(0, (new_row_2, new_col_2))
        else:
            nodes.append((new_row_2, new_col_2))
            nodes.insert(0, (new_row_1, new_col_1))
    
    if len(nodes) > 0:
        x, y = nodes[0]
    else:
        x, y = (False, False)

    if width % 2 == 0:
        new_row = row + (side + 1) * dir_x
        new_col = column + (side + 1) * dir_y
        #print("__________", row, new_row, column, new_col,"_________")
        if is_valid(row=new_row, col=new_col, rows=rows, columns=columns) and \
            is_unblocked(array = grid, row=new_row, col=new_col, values=values):
                if x == False or (new_row > x or (new_row == x and new_col == y)):
                    nodes.append((new_row, new_col))
                else:
                    nodes.insert(0, (new_row, new_col))
        else:
            new_row = row + (side + 1) * (-dir_x)
            new_col = column + (side + 1) * (-dir_y) # 2
            if is_valid(row=new_row, col=new_col, rows=rows, columns=columns) and \
                is_unblocked(array = grid, row=new_row, col=new_col, values=values):
                if x == False or (new_row > x or (new_row == x and new_col == y)):
                    nodes.append((new_row, new_col))
                else:
                    nodes.insert(0, (new_row, new_col))
    return nodes


def get_adjent_path(grid, path, width: int, values):
    '''
    Returns an array containg other nodes adjent to main path, according to path's width
    Parameters:
    grid    (array(int)): 
    path     (list): list consisting of (X, Y) coordinates for a given path
    width     (int): width of the path
    value     (int):
    '''
    other_nodes = []
    other_nodes.append([]) # start

    for i in range(1, len(path) - 1):
        prev_row, prev_column = path[i-1]
        current_row, current_column = path[i]
        dir_x, dir_y = get_offset(current_row=current_row, current_column=current_column,
                                  previous_row=prev_row, previous_column=prev_column)
        dir_x_perp, dir_y_perp = get_perpendicular_direction(dir_x=dir_x, dir_y=dir_y)
        neighbors = get_nodes(grid=grid, row=current_row, column=current_column,
                            width=width, dir_x=dir_x_perp, dir_y=dir_y_perp, values=values)
        other_nodes.append(neighbors)
    
    other_nodes.append([]) # end
    
    return other_nodes


# TODO
def assign_values_to_pads(grid, routes, pads):
    '''
    For compatibily of complex paths (paths made from 2 routes between 3 pads)
    '''
    values = []
    return values


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



def check_3x3_square(arr, row, column, rows: int, columns: int):
    for j in range(-1, 2):
        for k in range(-1, 2):
            aux_x = row + j
            aux_y = column + k
            if is_valid(row = aux_x, col = aux_y, rows = rows, columns = columns) and \
                not is_unblocked(array = arr, row = aux_x, col = aux_y, values=[0]):
                    return False
    return True


def check_line(arr, row: int, col: int, sign_x: int, sign_y: int,
                rows: int, columns: int, side: int, value):
    for i in range(1, side + 1):
        new_row = row + i * sign_x
        new_col = col + i * sign_x
        if not is_valid(row=new_row, col=new_col, rows=rows, columns=columns) or \
            not is_unblocked(array = arr, row=new_row, col=new_col, values=value):
                return False
        new_row = row + i * (-sign_x)
        new_col = col + i * (-sign_y)
        if not is_valid(row=new_row, col=new_col, rows=rows, columns=columns) or \
            not is_unblocked(array = arr, row=new_row, col=new_col, values=value):
                return False
    return True


# Functions (bool type return) that uses a grid to determine for a cell coordinate if there is enough space for a route to be place 
# (according to path's width and clearance)
def check_width_and_clearance(array, row: int, col: int, dir_x: int, dir_y: int, 
                              path_values: int, width: int = 1, clearance: int = 1):
    '''
    Functions (bool type return) that uses a grid to determine for a cell coordinate if there is enough 
    space for a route to be placed (according to path's width and clearance).
    Parameters:
        grid    (array): used to check for value; stores int data type
        row       (int): X coord for path's center cell
        ol       (int): Y coord for path's center cell
        dir_x     (int): direction on X: -1, 0, 1
        dir_y     (int): direction on Y: -1, 0, 1
        width     (int): path's width; path is divided in 3 types of path's: center: w = 1, left: w = width/2 (-1 if even), right: w = width/2
        clearance (int): space needed by path to not be already used
    '''
    # [(P.x, P.y), (P.x + dir_x, P.y + dir_y)] is perpendicular to [(P.x, P.y), (P.x + dir_perp_x, P.y + dir_perp_y)]
    dir_perp_x, dir_perp_y = get_perpendicular_direction(dir_x = dir_x, dir_y = dir_y)
    rows = len(array)
    columns = len(array[0])
    side = (width - 1) // 2
    add_dir = True

    value = path_values
    if dir_x == 0 :
        if not check_line(arr=array, row=row, col=col, rows=rows, columns=columns,
                            sign_x=1, sign_y=0, side=side+1, value=value):
            return False
    elif dir_y == 0:
        if not check_line(arr=array, row=row, col=col, rows=rows, columns=columns,
                            sign_x=0, sign_y=1, side=side+1, value=value):
            return False     
    else:
        if dir_x == dir_y: # -1,-1 ; 1,1
            if not check_line(arr=array, row=row, col=col, rows=rows, columns=columns,
                              sign_x=-1, sign_y=1, side=side+1, value=value):
                return False
            if not check_line(arr=array, row=row, col=col, rows=rows, columns=columns,
                              sign_x=1, sign_y=-1, side=side+1, value=value):
                return False
        else: # -1,1 ; 1;-1
            if not check_line(arr=array, row=row, col=col, rows=rows, columns=columns,
                sign_x=-1, sign_y=-1, side=side+1, value=value):
                return False
            if not check_line(arr=array, row=row, col=col, rows=rows, columns=columns,
                sign_x=1, sign_y=1, side=side+1, value=value):
                return False
        
        if not check_line(arr=array, row=row+dir_x, col=col+dir_y, rows=rows, columns=columns,
                sign_x=dir_perp_x, sign_y=dir_perp_y, side=side, value=value):
            return False
        if not check_line(arr=array, row=row+dir_x, col=col+dir_y, rows=rows, columns=columns,
                sign_x=-dir_perp_x, sign_y=-dir_perp_y, side=side, value=value):
            return False

    if width % 2 == 0: # asymetric case
        new_row = row + dir_x + (side + 2) * dir_perp_x
        new_col = col + dir_y + (side + 2) * dir_perp_y
        if not is_valid(row=new_row, col=new_col, rows=rows, columns=columns) or \
            not is_unblocked(array = array, row=new_row, col=new_col, values=value):
                add_dir = False
                new_row = row + dir_x - (side + 2) * dir_perp_x
                new_col = col + dir_y - (side + 2) * dir_perp_y
                if not is_valid(row=new_row, col=new_col, rows=rows, columns=columns) or \
                    not is_unblocked(array = array, row=new_row, col=new_col, values=value):
                        return False

    for i in range(0, clearance):
        if width % 2 == 1:
            new_x = row + dir_x + (side + i) * dir_perp_x
            new_y = col + dir_y + (side + i) * dir_perp_y
            if not check_3x3_square(arr = array, row = new_x, column = new_y, rows = rows, columns = columns):
                return False
            
            new_x = row + dir_x - (side + i) * dir_perp_x
            new_y = col + dir_y - (side + i) * dir_perp_y
            if not check_3x3_square(arr = array, row = new_x, column = new_y, rows = rows, columns = columns):
                return False
                
        else:
            if add_dir == True:
                new_x = row + dir_x + (side + i) * dir_perp_x
                new_y = col + dir_y + (side + i) * dir_perp_y
                if not check_3x3_square(arr = array, row = new_x, column = new_y, rows = rows, columns = columns):
                    return False
                
                new_x = row + dir_x - (side + i - 1) * dir_perp_x
                new_y = col + dir_y - (side + i - 1) * dir_perp_y
                if not check_3x3_square(arr = array, row = new_x, column = new_y, rows = rows, columns = columns):
                    return False

            else:
                new_x = row + dir_x - (side + i - 1) * dir_perp_x
                new_y = col + dir_y - (side + i - 1) * dir_perp_y
                if not check_3x3_square(arr = array, row = new_x, column = new_y, rows = rows, columns = columns):
                    return False
                
                new_x = row + dir_x + (side + i) * dir_perp_x
                new_y = col + dir_y + (side + i) * dir_perp_y
                if not check_3x3_square(arr = array, row = new_x, column = new_y, rows = rows, columns = columns):
                    return False

    return True



# find one route at a time using A star algorihm (modified Dijkstra)
def a_star_search(grid, grid_size: tuple, st_row: int, st_col: int, dest_row: int, dest_col: int, path_value: int,
                  closed_array, cell_details, clearance: int = 1, width: int = 1, hide_prints = True):
    rows, columns = grid_size

    if not is_valid(st_row, st_col, rows, columns) or not is_valid(dest_row, dest_col, rows, columns):
        if hide_prints == False:
            print("\nStart | Dest invalid")
        return False
    
    if not is_unblocked(grid, st_row, st_col, values=[0, path_value]) or not is_unblocked(grid, dest_row, dest_col, values=[0, path_value]):
        if hide_prints == False:
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

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while len(open_list) > 0:
        p = heapq.heappop(open_list)
        # Mark cell as visited
        i = p[1]
        j = p[2]
        closed_array[i][j] = True

        # for each direction check the succesor
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            if is_valid(row = new_i, col = new_j, rows = rows, columns = columns) and is_unblocked(grid, new_i, new_j, values=[0, path_value, path_value + 0.5]) and \
                not closed_array[new_i][new_j]:

                if check_width_and_clearance(array = grid, path_values = [0, path_value, path_value + 0.5], row = i, col = j,
                                              dir_x = dir[0], dir_y = dir[1], width = width, clearance = clearance):
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
                            if hide_prints == False:
                                print("\n\nDestination cell reached")
                            return True

    if hide_prints == False:
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



# TODO: routing with lee
def lee_search(grid, grid_size: tuple, st_row: int, st_col: int, dest_list: list,
               closed_array, width: int, clearance: int, values: list, hide_prints: bool = True):
    '''
    grid    (array):
    grid_size ((int, int)):
    st_row  (int):
    st_col  (int):
    dest    (list((int, int)):
    closed_array (array):
    values  (list(int)):
    hide_prints (bool):
    '''

    rows, columns = grid_size
    if not is_valid(row=st_row, col=st_col, rows=rows, columns=columns):
        if hide_prints == False:
            print("\nStart invalid")
        return False

    if not is_unblocked(array = grid, row = st_row, col = st_col, values = values):
        if hide_prints == False:
            print("\nStart blocked")
        return False
    
    for dest in dest_list:
        row, col = dest
        if not is_valid(row=row, col=col, rows=rows, columns=columns):
            if hide_prints == False:
                print("\nAt least one destination is invalid")
            return False

        if not is_unblocked(array = grid, row = row, col = row, values = values):
            if hide_prints == False:
                print("\nAt least one destination is blocked")
            return False

    # Create a queue for BFS 
    q = deque()

    # initialize start of the list
    i = st_row
    j = st_col
    closed_array[i][j] = True
    cost = 0
    s = (i, j, 0.0)
    q.append(s)

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while len(q) > 0:
        p = q.popleft()
        # Mark cell as visited
        i, j, cost = p
        
        if grid[i][j] in values:
            for dest in dest_list:
                x, y = dest
                if is_destination(row = i, col = j, dest_row = x, dest_col = y):
                    if hide_prints == False:
                        print(f"\n\nDestination (pad) ({i}, {j}) reached")
                    return True
            print(f"\n\nDestination (route) ({i}, {j}) reached")
            return True

        # for each direction check the succesor
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            if is_valid(row = new_i, col = new_j, rows = rows, columns = columns) and \
                is_unblocked(array = grid, row = new_i, col = new_j, values = values) and \
                not closed_array[new_i][new_j]:

                if check_width_and_clearance(array = grid, path_values = values, row = i, col = j,
                                              dir_x = dir[0], dir_y = dir[1], width = width, clearance = clearance):
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
                        closed_array[new_i][new_j] = True
                        new_cost = cost + h_euclidian(st_row = i, st_col = j, dest_row = new_i, dest_col = new_j)
                        q.append(new_i, new_j, new_cost)

    if hide_prints == False:
        print("\nDestination not reached")
    
    return False





# run multiple A* searches

'''
# add option for appending to existing path
'''
def get_paths(grid, routes: list = None, pads: list = None, existing_paths = None, width_list: list = None, 
              clearance_list: int = 2, draw_flag = False, hide_prints = True, color_matrix = None):
    '''
    Returns grid used during process and paths found as a list(Path)
    Parameters:
    grid                     (array(int)):
    routes                         (list):
    pads                      (list(Pad)):
    existing_paths    (list(Path) | None):
    width_list                (list(int)):
    color_matrix           (array | None):
    clearance_list            (list(int)):
    draw_flag                      (bool):
    hide_prints_flag               (bool):
    '''
    rows = len(grid)
    columns = len(grid[0])
    
    n = len(routes)
    if not width_list or len(width_list) == 0:
        width_list = [1 for i in range(n)]
        width_list[0] = 3
        width_list[1] = 2

    placed_points = []
    if existing_paths:
        paths = copy.deepcopy(existing_paths)
        for path in existing_paths:
            placed_points.append([path[0], path[len(path) - 1]])
    else:
        paths = []

    route_index = 0
    x = np.array(routes).shape
    grid_copy = copy.deepcopy(grid)
    if x != (4,): # at least 2 routes
        closed_array = [[False for _ in range(columns)] for _ in range(rows)] # visited cells

        for pad in pads:
            area = pad.occupied_area
            for coord in area:
                coord_x, coord_y = coord
                grid_copy[coord_x][coord_y] = -1

        for route in routes:    
            cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)] # status of every cell in the grid
            route_index += 1
            st_row, st_col, dest_row, dest_col = route[0:4]

            # 0 = unblock; -1  = blocked; 1,2,... = used

            pad_area1 = []
            pad_area2 = []

            for pad in pads:
                coord_x = pad.center_x
                coord_y = pad.center_y
                if (st_row == coord_x and st_col == coord_y) or (dest_row == coord_x and dest_col == coord_y):
                    area = pad.occupied_area
                    for coord in area:
                        coord_x, coord_y = coord
                        grid_copy[coord_x][coord_y] = 0
                        if st_row == pad.center_x:
                            pad_area1.append((coord_x, coord_y))
                        else:
                            pad_area2.append((coord_x, coord_y))            

            result = a_star_search(grid = grid_copy, grid_size = (rows, columns), path_value = route_index,
                                   st_row = st_row, st_col = st_col, dest_row = dest_row, dest_col = dest_col,
                                   width = width_list[route_index-1], clearance = clearance_list[route_index-1],
                                   closed_array = closed_array, cell_details = cell_details, hide_prints = hide_prints)
            
            if result == True:
                path = get_path_A_star(cell_details = cell_details, dest_row = dest_row, dest_col = dest_col)
                adjent_path = get_adjent_path(grid=grid_copy, path=path, width=width_list[route_index-1], values=[0, route_index])

                path_found = Path(start_x = st_row, start_y = st_row, dest_x = dest_row, dest_y = dest_col, 
                                  width = width_list[route_index-1], clearance = clearance_list[route_index-1],
                                  path = path, other_nodes = adjent_path)

                if hide_prints == False:
                    print_path(path = path_found.path, path_index = route_index)
                
                paths.append(path_found)

                # TODO remove paths, add function for marking 
                for cell in path:
                    i, j = cell
                    grid_copy[i][j] = route_index

                extended_path = []
                if len(adjent_path) > 0:
                    for subpath in adjent_path:
                        extended_path.extend(subpath)
                        for cell in subpath:
                            i, j = cell
                            grid_copy[i][j] = route_index + 0.5


                for cell in pad_area1:
                    i, j = cell
                    if cell in path:
                        grid_copy[i][j] = route_index
                    else:
                        grid_copy[i][j] = -1

                for cell in pad_area2:
                    i, j = cell
                    if cell in path:
                        grid_copy[i][j] = route_index
                    else:
                        grid_copy[i][j] = -1
                # --- 

                if draw_flag == True:    draw_grid(color_matrix = color_matrix, 
                                              main_path = path, color_main_path = random.choice([COLORS['yellow'], COLORS['red'], COLORS['pink'], COLORS['orange']]),
                                              other_nodes = extended_path, color_other_nodes = random.choice([COLORS['yellow'], COLORS['red'], COLORS['pink'], COLORS['orange']])) 

                
                # mark grid cells - for lee algorithm
                set_values_in_array(blocked_cells = extended_path, arr = grid_copy, value = True)

                extended_path.extend(path)
                reset_cells_in_array(array = closed_array, path = extended_path, reset_value = False)

            else:
                if hide_prints == False:
                    print("\tNo change in drawing. Route can't be placed\n")
                reset_cells_in_array(array = closed_array, path = None, reset_value = False)
                paths.append(Path(start_x = st_row, start_y = st_col, dest_x = dest_row, dest_y = dest_col, 
                                  width = width_list[route_index-1], clearance=clearance_list[route_index-1],
                                  path = [], other_nodes = []))
    
    else: # one single route
        route_index = 1
        st_row, st_col, dest_row, dest_col = route[0:4]

        for pad in pads:
            coord_x = pad.center_x
            coord_y = pad.center_y
            if (st_row == coord_x and st_col == coord_y) or (dest_row == coord_x and dest_col == coord_y):
                area = pad.occupied_area
                for coord in area:
                    coord_x, coord_y = coord
                    grid_copy[coord_x][coord_y] = 0
                    if st_row == coord_x:
                        pad_area1.append((coord_x, coord_y))
                    else:
                        pad_area2.append((coord_x, coord_y))

        result = a_star_search(grid = grid_copy, grid_size = (rows, columns), path_value = route_index,
                               st_row = st_row, st_col = st_col, dest_row = dest_row, dest_col = dest_col,
                               width = width_list[0], clearance = clearance_list[0], 
                               closed_array = closed_array, cell_details = cell_details, hide_prints = hide_prints)

        if result:
            path = get_path_A_star(cell_details = cell_details, dest_row = dest_row, dest_col = dest_col, suppress_prints = hide_prints)
            
            adjent_path = get_adjent_path(grid=grid_copy, path=path, width=width, values=[0, route_index])
            path_found = Path(start_x = st_row, start_y = st_row, dest_x = dest_row, dest_y = dest_col, 
                              width = width_list[0], clearance = clearance_list[0],
                              path = path, other_nodes = adjent_path)
            
            if hide_prints == False:
                print_path(path = path_found.path, path_index = route_index)
            paths.append(path_found)

            for cell in path:
                i, j = cell
                grid_copy[i][j] = route_index

            extended_path = []
            if len(adjent_path) > 0:
                for subpath in adjent_path:
                    extended_path.extend(subpath)
            
            # --
            if len(extended_path) > 0: 
                for cell in extended_path:
                    i, j = cell 
                    grid_copy[i][j] = route_index

            for cell in pad_area1:
                i, j = cell
                if cell in path:
                    grid_copy[i][j] = route_index
                else:
                    grid_copy[i][j] = -1

            for cell in pad_area2:
                i, j = cell
                if cell in path:
                    grid_copy[i][j] = route_index
                else:
                    grid_copy[i][j] = -1
            # --- 

            if draw_flag == True:    draw_grid(color_matrix = color_matrix, 
                                            main_path = path, color_main_path = random.choice([COLORS['yellow'], COLORS['red'], COLORS['pink'], COLORS['orange']]),
                                            other_nodes = extended_path, color_other_nodes = random.choice([COLORS['yellow'], COLORS['red'], COLORS['pink'], COLORS['orange']])) 
        
            # mark grid cells - for lee algorithm
            set_values_in_array(blocked_cells = extended_path, arr = grid_copy, value = True)

            extended_path.extend(path)
            reset_cells_in_array(array = closed_array, path = extended_path, reset_value = False)
    
        else:
            if hide_prints == False:
                print("\tNo change in drawing. Route can't be placed\n")
            paths.append(Path(start_x = st_row, start_y = st_col, dest_x = dest_row, dest_y = dest_col,
                              width = width_list[0], clearance = clearance_list[0],
                              path = [], other_nodes = []))
    # unplaced_routes to be changed to sth to check if there are all nodes present in solution and if not, counts the remaining ones

    return grid_copy, paths



# check if initial representation is needed and start placing routes
def solution(grid, routes, pads, clearance_list: list = None, width_list: list = None,
              rows = ROWS, columns = COLS, blocked_areas = None, colors = None, draw = False):
    if draw == True:
        color_matrix = get_RGB_matrix(nodes = routes, colors_list = colors, background = COLORS['black'], rows = rows, columns = columns)
        #color_matrix = mark_areas_in_grid(blocked_cells = blocked_areas, grid = color_matrix, value = COLORS['red'])
        color_matrix = color_pads_in_RGB_matrix(pads = pads, rows = rows, columns = columns, grid = color_matrix, background = COLORS['black'])
        draw_grid(color_matrix = color_matrix, main_path=None)

    clearance_list = [1 for i in range(len(routes))]
    grid_copy, paths = get_paths(grid = grid, routes = routes,
                                pads = pads, clearance_list = clearance_list,
                                color_matrix = color_matrix, draw_flag = draw, 
                                hide_prints = False)
    



# for testing purposes
if __name__ == "__main__":
    rows = ROWS
    columns = COLS
    pins_sizes = 3
    pads = []
    #rectangle = generate_rectangle(row = 10, col = 0, length_x = 1, length_y = 2)
    #blocked  = [(4,0), (4,4), (4,8), (10,10), (10,1), (2,4), (1,4), (1,5)]
    #blocked = list(set(blocked + rectangle)) # remove duplicates
    blocked = None

    routes, colors = read_file_routes(file_name='pins.csv', draw = True)

    # for testing Pad class
    for route in routes:
        if len(pads) != 0:
            pin = Pad(center_x = route[0], center_y = route[1], length = 1, width = 1, occupied_area = [(route[0], route[1])])
            if pin not in pads:
                pads.append(pin)
            pin = Pad(center_x = route[2], center_y = route[3], length = 1, width = 1, occupied_area = [(route[2], route[3])])
            if pin not in pads:
                pads.append(pin)
        else:
            height = 3
            width = 3
            occupied_area = []
            pad = Pad(center_x = route[0], center_y = route[1], length = height, width = width, occupied_area = [(route[0], route[1])])
            for h in range(height):
                for w in range(width):
                    x = pad.center_x - height // 2 + h 
                    y = pad.center_y - width // 2 + w
                    occupied_area.append((x,y))
            pad.occupied_area = occupied_area
            pads.append(pad)

            occupied_area = []
            pad = Pad(center_x = route[2], center_y = route[3], length = height, width = width, occupied_area = [(route[2], route[3])])
            for h in range(height):
                for w in range(width):
                    x = pad.center_x - height // 2 + h
                    y = pad.center_y - width // 2 + w
                    occupied_area.append((x,y))
            pad.occupied_area = occupied_area
            pads.append(pad)


    grid = np.zeros((rows, columns), dtype=float)
    n = len(routes)
    widths = [1 for i in range(n)]
    widths[0] = 3
    widths[3] = 2
    widths[2] = 3

    # mark with red cells that can be used (obstacles)
    grid = set_values_in_array(blocked_cells = blocked, arr = grid, value = -1)
    for area in pads:
        print(area.occupied_area)
    print('\n')
    solution(grid = grid, routes = routes, pads = pads, width_list = widths,
             colors = colors, blocked_areas = blocked, draw = True)

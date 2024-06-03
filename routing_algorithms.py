import numpy as np # --
import heapq
from collections import deque
import copy
import random
from colored_repr_utils import draw_grid, get_RGB_matrix, color_pads_in_RGB_matrix, COLORS
from utils import Cell, Path, Pad, is_destination, is_unblocked, is_valid, h_euclidian, \
                  read_file_routes, set_area_in_array, check_element_in_list, \
                  get_perpendicular_direction
#from pcb_utils import PlannedRoute

# Define size of grid
ROWS = 55
COLS = 55


# equivalent to mark_apth_in)grid from pcb_utils
def set_values_in_array(blocked_cells, arr, value: 0): 
    # value = 1, 2, ... for routes, 0 for other types of obstacle
    arr_copy = copy.deepcopy(arr)
    if blocked_cells:
        for cell in blocked_cells:
            i, j = cell
            arr_copy[i][j] = value
    return arr_copy


def get_offset_for_routing(current_poz: tuple[int, int], previous_poz: tuple[int, int]):
    '''Returns distance for X and Y between 2 points reprezented as (int, int)'''
    current_row, current_column = current_poz
    previous_row, previous_column = previous_poz
    return current_poz[0] - previous_poz[0], current_poz[1] - previous_poz[1]



def get_nodes(grid, point: tuple[int, int], width: int, direction_y: int, direction_x: int, values: int):
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
    row, column = point
    side = (width - 1) // 2
    for i in range(1, side + 1):
        new_row_1 = row + i * direction_y
        new_col_1 = column + i * direction_x
      
        new_row_2 = row - i * direction_y
        new_col_2 = column - i * direction_x

        if new_row_1 > new_row_2 or (new_row_1 == new_row_2 and new_col_1 > new_col_2):
            nodes.append((new_row_1, new_col_1))
            nodes.insert(0, (new_row_2, new_col_2))
        else:
            nodes.append((new_row_2, new_col_2))
            nodes.insert(0, (new_row_1, new_col_1))
    
    if len(nodes) > 0: # side > 1
        y, x = nodes[0]
    else:
        y, x = (False, False)

    if width % 2 == 0: # asymetric case; side widths: n, n+1
        new_row = row + (side + 1) * direction_y
        new_col = column + (side + 1) * direction_x
        if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), values):
                if x == False or (new_row > y or (new_row == y and new_col == x)):
                    nodes.append((new_row, new_col))
                else:
                    nodes.insert(0, (new_row, new_col))
        else:
            new_row = row - (side + 1) * direction_y
            new_col = column - (side + 1) * direction_x # 2
            if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), values):
                if y == False or (new_row > y or (new_row == y and new_col == x)):
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
    #other_nodes.append([]) # start;     also for end
    for i in range(1, len(path) - 1):
        previous_row, previous_column = path[i - 1]
        current_row, current_column = path[i]
        direction_y, direction_x = get_offset_for_routing(current_poz = (current_row, current_column), 
                                                            previous_poz = (previous_row, previous_column))
        direction_perp_y, direction_perp_x = get_perpendicular_direction(direction_y, direction_x) # direction perpendicular to x and y
        neighbors = get_nodes(grid = grid, point = (current_row, current_column), width = width,
                               direction_y = direction_perp_y, direction_x = direction_perp_x, values = values)
        other_nodes.append(neighbors)
    
    return other_nodes


# TODO
def assign_values_to_pads(grid, routes, pads):
    '''
    For compatibily of complex paths (paths made from 2 routes between 3 pads)
    '''
    values = []
    return values


def print_path(path, path_index = 0):
    if path_index > 0:
        message = f"\nPath {path_index}. is:"
        print(message)
        for i in path:
            print(i, end="->")



def check_3x3_square(array, point: tuple[int, int], array_shape: tuple[int, int]):
    rows, columns = array_shape
    current_row, current_column = point
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),  (1, 1)]
    
    for j, k in directions:
        neighbor_row = current_row + j
        neighbor_column = current_column + k
        
        if is_valid((neighbor_row, neighbor_column), (rows, columns)) and not is_unblocked(array, (neighbor_row, neighbor_column), [0]):
            return False
    
    return True


def check_line(array, point: tuple[int, int], sign_y: int, sign_x: int, array_shape: tuple[int, int], offset: int, value):
    '''side (int) - check side amount of values next to (current_row, current_column)'''
    current_row, current_column = point
    rows, columns = array_shape
    for i in range(1, offset + 1):
        new_row = current_row + i * sign_y
        new_col = current_column + i * sign_x
        if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), value):
                return False
    
        new_row = current_row - i * sign_y
        new_col = current_column - i * sign_x
        if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), value):
                return False
    return True



def check_width_and_clearance(array, array_shape: tuple[int, int], point: tuple[int, int], direction_y: int, direction_x: int, 
                              path_values: int, width: int = 1, clearance: int = 1):
    '''
    Functions (bool type return) that uses a grid to determine for a cell coordinate if there is enough 
    space for a route to be placed (according to path's width and clearance).
    Parameters:
        array      (array): used to check for value; stores int data type
        array_shape(tuple): shape of the array
        point     (tuple): (row, column) coordinates of the point to check
        direction_y (int): direction on Y: -1, 0, 1
        direction_x (int): direction on X: -1, 0, 1
        path_values (int): value of the path
        width      (int): path's width; path is divided in 3 types of path's: center: w = 1, left: w = width/2 (-1 if even), right: w = width/2
        clearance  (int): space needed by path to not be already used
    '''
    rows, columns = array_shape
    row, column = point
    path_side_width = (width - 1) // 2
    direction_perp_y, direction_perp_x = get_perpendicular_direction(direction_y, direction_x)
    value = path_values
    add_dir = True
    def check_main_lines():
        if direction_y == 0:
            return check_line(array, (row, column), 0, 1, (rows, columns), path_side_width + 1, value)
        elif direction_x == 0:
            return check_line(array, (row, column), 1, 0, (rows, columns), path_side_width + 1, value)
        elif direction_x == direction_y:
            return (check_line(array, (row, column), 1, -1, (rows, columns), path_side_width + 1, value) and
                    check_line(array, (row, column), -1, 1, (rows, columns), path_side_width + 1, value))
        else:
            return (check_line(array, (row, column), -1, -1, (rows, columns), path_side_width + 1, value) and
                    check_line(array, (row, column), 1, 1, (rows, columns), path_side_width + 1, value))

    def check_clearance_lines():
        for i in range(0, clearance):
            if width % 2 == 1:
                if (not check_3x3_square(array, (row + direction_y + (path_side_width + i) * direction_perp_y,
                                                 column + direction_x + (path_side_width + i) * direction_perp_x), (rows, columns)) or
                    not check_3x3_square(array, (row + direction_y - (path_side_width + i) * direction_perp_y,
                                                 column + direction_x - (path_side_width + i) * direction_perp_x), (rows, columns))):
                    return False
            else:
                if add_dir:
                    if (not check_3x3_square(array, (row + direction_y + (path_side_width + i) * direction_perp_y,
                                                     column + direction_x + (path_side_width + i) * direction_perp_x), (rows, columns)) or
                        not check_3x3_square(array, (row + direction_y - (path_side_width + i - 1) * direction_perp_y,
                                                     column + direction_x - (path_side_width + i - 1) * direction_perp_x), (rows, columns))):
                        return False
                else:
                    if (not check_3x3_square(array, (row + direction_y - (path_side_width + i - 1) * direction_perp_y,
                                                     column + direction_x - (path_side_width + i - 1) * direction_perp_x), (rows, columns)) or
                        not check_3x3_square(array, (row + direction_y + (path_side_width + i) * direction_perp_y,
                                                     column + direction_x + (path_side_width + i) * direction_perp_x), (rows, columns))):
                        return False
        return True

    def check_even_width_asymmetry():
        if width % 2 == 0:
            new_row = row + direction_y + (path_side_width + 2) * direction_perp_y
            new_col = column + direction_x + (path_side_width + 2) * direction_perp_x
            if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), value):
                add_dir = False
                new_row = row + direction_y - (path_side_width + 2) * direction_perp_y
                new_col = column + direction_x - (path_side_width + 2) * direction_perp_x
                if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), value):
                    return False
        return True

    if not check_main_lines():
        return False
    if not check_even_width_asymmetry():
        return False
    if not check_clearance_lines():
        return False

    return True


# either this or the one above

# def check_width_and_clearance(array, array_shape: tuple[int, int], point: tuple[int, int], direction_y: int, direction_x: int, 
#                               path_values: int, width: int = 1, clearance: int = 1):
#     '''
#     Functions (bool type return) that uses a grid to determine for a cell coordinate if there is enough 
#     space for a route to be placed (according to path's width and clearance).
#     Parameters:
#         array      (array): used to check for value; stores int data type
#         array_shape(tuple): shape of the array
#         point     (tuple): (row, column) coordinates of the point to check
#         direction_y (int): direction on Y: -1, 0, 1
#         direction_x (int): direction on X: -1, 0, 1
#         path_values (int): value of the path
#         width      (int): path's width; path is divided in 3 types of path's: center: w = 1, left: w = width/2 (-1 if even), right: w = width/2
#         clearance  (int): space needed by path to not be already used
#     '''
#     rows, columns = array_shape
#     row, column = point
#     path_side_width = (width - 1) // 2
#     direction_perp_y, direction_perp_x = get_perpendicular_direction(direction_y, direction_x)
#     value = path_values
#     add_dir = True
#     def check_lines(offset):
#         if direction_y == 0:
#             return check_line(array, (row, column), 0, 1, (rows, columns), offset, value)
#         elif direction_x == 0:
#             return check_line(array, (row, column), 1, 0, (rows, columns), offset, value)
#         elif direction_x == direction_y:
#             return (check_line(array, (row, column), 1, -1, (rows, columns), offset, value) and
#                     check_line(array, (row, column), -1, 1, (rows, columns), offset, value))
#         else:
#             return (check_line(array, (row, column), -1, -1, (rows, columns), offset, value) and
#                     check_line(array, (row, column), 1, 1, (rows, columns), offset, value))

#     def check_clearance_margins():
#         for i in range(width + clearance):
#             if width % 2 == 1:
#                 if not check_3x3_square(array, (row + direction_y + i * direction_perp_y, column + direction_x + i * direction_perp_x), (rows, columns)):
#                     return False
#                 if not check_3x3_square(array, (row + direction_y - i * direction_perp_y, column + direction_x - i * direction_perp_x), (rows, columns)):
#                     return False
#             else:
#                 if add_dir:
#                     if not check_3x3_square(array, (row + direction_y + i * direction_perp_y, column + direction_x + i * direction_perp_x), (rows, columns)):
#                         return False
#                     if not check_3x3_square(array, (row + direction_y - (i - 1) * direction_perp_y, column + direction_x - (i - 1) * direction_perp_x), (rows, columns)):
#                         return False
#                 else:
#                     if not check_3x3_square(array, (row + direction_y - (i - 1) * direction_perp_y, column + direction_x - (i - 1) * direction_perp_x), (rows, columns)):
#                         return False
#                     if not check_3x3_square(array, (row + direction_y + i * direction_perp_y, column + direction_x + i * direction_perp_x), (rows, columns)):
#                         return False
#         return True

#     def check_even_width_asymmetry():
#         if width % 2 == 0:
#             new_row = row + direction_y + (path_side_width + 2) * direction_perp_y
#             new_col = column + direction_x + (path_side_width + 2) * direction_perp_x
#             if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), value):
#                 add_dir = False
#                 new_row = row + direction_y - (path_side_width + 2) * direction_perp_y
#                 new_col = column + direction_x - (path_side_width + 2) * direction_perp_x
#                 if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), value):
#                     return False
#         return True

#     if not check_lines(path_side_width + clearance):
#         return False
#     if not check_even_width_asymmetry():
#         return False
#     if not check_clearance_margins():
#         return False

#     return True




# find one route at a time using A star algorihm (modified Dijkstra)
def a_star_search(grid, grid_size: tuple[int, int], start: tuple[int, int], destination: tuple[int, int], path_value: int,
                  clearance: int = 1, width: int = 1, hide_prints = True):
    rows, columns = grid_size
    start_row, start_col = start
    destination_row, destiantion_col = destination
    
    if not is_valid((start_row, start_col), (rows, columns)) or not is_valid((destination_row, destiantion_col), (rows, columns)):
        if hide_prints == False:
            print("\nStart | Dest invalid")
        return False
    
    if not is_unblocked(grid, (start_row, start_col), values = [0, path_value]) or not \
        is_unblocked(grid, (destination_row, destiantion_col), values = [0, path_value]):
        if hide_prints == False:
            print("\nStart | Dest blocked")
        return False

    # Return the path from source to destination
    def get_path_A_star():
        path = []
        row, column = destination
    
        while not (cell_details[row][column].parent == (row, column)):
            path.append((row, column))
            row, column = cell_details[row][column].parent
        
        path.append((row, column)) # add start node to path
        path.reverse()
            
        return path

    # initialize start of the list
    i = start_row
    j = start_col
    
    cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)] # status of every cell in the grid
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent = (i, j)

    open_list = []  # cells to be visited
    heapq.heappush(open_list, (0.0, i, j))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
  
    while len(open_list) > 0:
        point = heapq.heappop(open_list)
        i, j = point[1], point[2]
        print("Progress:", i, j, ":", start_row, start_col, ":", destination_row, destiantion_col)

        for dir in directions:  # for each direction check the succesor
            dir_y, dir_x = dir
            new_i = i + dir_y
            new_j = j + dir_x

            if is_valid((new_i, new_j), (rows, columns)) and \
                is_unblocked(grid, (new_i, new_j), [0, path_value, path_value + 0.5]) and \
                  cell_details[new_i][new_j].parent == (-1, -1):


                if check_width_and_clearance(grid, (rows, columns), (i, j), dir_y, dir_x, [0, path_value, path_value + 0.5], width, clearance) \
                    and not is_destination((new_i, new_j), (destination_row, destiantion_col)):
                    parent_cell = cell_details[i][j]
                    nr_bends = parent_cell.bends
                    nr_90_degree_bends = parent_cell.bends_90_deg
                    parent_direction = parent_cell.direction
                    if parent_direction and parent_direction != dir:
                        nr_bends += 1
                        aux_y, aux_x = parent_cell.parent
                        if (aux_y, aux_x) != (-1, -1):
                            aux_direction = cell_details[aux_y][aux_x].direction
                            if aux_direction and parent_direction != aux_direction:
                                nr_90_degree_bends += 1                                    

                    g_new = parent_cell.g + h_euclidian((i, j), (new_i, new_j))   # greedy aprouch
                    h_new = h_euclidian((new_i, new_j), (destination_row, destiantion_col)) * 10
                    f_new = g_new + h_new + (nr_bends << 2) + (nr_90_degree_bends << 4)

                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].bends = nr_bends
                        cell_details[new_i][new_j].bends_90_deg = nr_90_degree_bends
                        cell_details[new_i][new_j].direction = dir
                        cell_details[new_i][new_j].parent = (i, j)

                elif is_destination((new_i, new_j), (destination_row, destiantion_col)):
                    cell_details[new_i][new_j].parent = (i, j)
                    if hide_prints == False:
                        print("\n\nDestination cell reached")

                    path = get_path_A_star()
                    return path

    if hide_prints == False:
        print("\nDestination not reached")
    return []



# routing with lee
def lee_search(grid, grid_size: tuple, start: tuple[int, int], possible_ends: list,
               width: int, clearance: int, value: int, hide_prints: bool = True):
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
    start_row, start_column = start

    if not is_valid((start_row, start_column), (rows, columns)):
        if hide_prints == False:
            print("\nStart invalid")
        return False
    
    values = [value]
    values.append(0)
    #values.append(value + 0.5)

    if not is_unblocked(grid, (start_row, start_column), values):
        if hide_prints == False:
            print("\nStart blocked")
        return False
    
    # for dest in possible_ends:
    #     row, column = dest
    #     if not is_valid((row, column), (rows, columns)):
    #         if hide_prints == False:
    #             print("\nAt least one destination is invalid")
    #         return False

    #     if not is_unblocked(grid_copy, (row, column), values):
    #         if hide_prints == False:
    #             print("\nAt least one destination is blocked")
    #         return False         

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    mark_path_in_array = np.full(shape = (rows, columns), fill_value = -1, dtype=float)
    dest_x, dest_y = (-1, -1)

    def get_lee_path():
        path = []
        mark_path_in_array[start_row][start_column] = 0

        previous_direction = None  # Inițializează direcția anterioară ca fiind None
        best_y, best_x = dest_y, dest_x
        best_value = mark_path_in_array[best_y][best_x]

        current_x, current_y = best_x, best_y

        while True:
            path.append((best_y, best_x))
            if best_value == 0:     # start point
                return path, (dest_y, dest_x)

            if previous_direction:
                dir_y, dir_x = previous_direction
                aux_y = current_y + dir_y
                aux_x = current_x + dir_x
                cost = mark_path_in_array[aux_y][aux_x]
                if best_value > cost and cost >= 0:
                    best_x, best_y = aux_x, aux_y
                    best_value = cost

            for dir in directions:
                dir_y, dir_x = dir
                aux_y = current_y + dir_y
                aux_x = current_x + dir_x

                cost = mark_path_in_array[aux_y][aux_x]
                if best_value > cost and cost >= 0:
                    best_x, best_y = aux_x, aux_y
                    previous_direction = dir
                    best_value = cost
            
            current_x, current_y = best_x, best_y

    end_reached = False
    q = deque() # Create a queue for BFS
    
    # Add start to BFS q
    i, j = start_row, start_column
    # visited = np.full((rows, columns), fill_value=False)
    # visited[i][j] = True
    visited = set()
    visited.add((i, j))
    s = (i, j, 0.0) 
    q.append(s) 
    
    while len(q) > 0:
        entry = q.popleft()
        i, j, cost = entry
        mark_path_in_array[i][j] = cost
        
        if i != start_row or j != start_column:         
            if grid[i][j] in values:
                for dest in possible_ends:
                    y, x = dest
                    if is_destination((i, j), (y, x)):
                        end_reached = True
                        if not hide_prints:
                            print(f"\n\nDestination (pad) ({i}, {j}) reached")
            elif grid[i][j] == value:
                end_reached = True
                if not hide_prints:
                    print(f"\n\nDestination (route) ({i}, {j}) reached")

            if end_reached:
                dest_y, dest_x = i, j
                path = get_lee_path()
                return path

        # for each direction check the succesor
        for dir in directions:
            dir_y, dir_x = dir
            new_i = i + dir_y
            new_j = j + dir_x

            if is_valid((new_i, new_j), (rows, columns)) and is_unblocked(grid, (new_i, new_j), values) and \
                 (new_i, new_j) not in visited:#visited[new_i][new_j] == False:

                if check_width_and_clearance(array = grid, array_shape = (rows, columns), 
                                             path_values = values, point = (i, j), direction_y = dir_y, 
                                             direction_x = dir_x, width = width, clearance = clearance):
                    visited.add((new_i, new_j)) #visited[new_i][new_j] = True
                    new_cost = cost + h_euclidian((i, j), (new_i, new_j))
                    q.append((new_i, new_j, new_cost))
    
    if hide_prints == False:
        print("\nDestination not reached")
    return [], (-1, -1)


def mark_clearance_on_grid(grid, grid_shape, path, path_width: int, clearance_width: int, clearance_value: float):
    """
    Marks the grid with clearance values around the given path.

    Parameters:
    grid            (array): The grid to be marked.
    path            (list) : List of (X, Y) coordinates for the main path.
    clearance_width (int)  : The width of the clearance to be added around the path.
    path_width      (int)  : The width of the path.
    clearance_value (int)  : The value to mark for clearance on the grid.
    """
    rows, columns = grid_shape
    path_side_width = (path_width - 1) // 2
    max_width = path_side_width + clearance_width + 1

    for i in range(1, len(path)-1):
        current_row, current_column = path[i]
        next_row, next_column = path[i + 1]
        
        direction_y, direction_x = next_row - current_row, next_column - current_column
        direction_perp_y, direction_perp_x = get_perpendicular_direction(direction_y, direction_x)

        for j in range(path_side_width + 1, max_width):
            new_row = current_row + j * direction_perp_y
            new_col = current_column + j * direction_perp_x
            if 0 <= new_row < rows and 0 <= new_col < columns:
                if grid[new_row, new_col] == 0:
                    grid[new_row, new_col] = clearance_value

            new_row = current_row - j * direction_perp_y
            new_col = current_column - j * direction_perp_x
            if 0 <= new_row < rows and 0 <= new_col < columns:
                if grid[new_row, new_col] == 0:
                    grid[new_row, new_col] = clearance_value

        # Handle the asymmetric case if path_width is even
        if path_width % 2 == 0:
            extra_row = current_row + (max_width + 1) * direction_perp_y
            extra_col = current_column + (max_width + 1) * direction_perp_x
            
            if 0 <= extra_row < rows and 0 <= extra_col < columns:
                if grid[extra_row, extra_col] == 0:
                    grid[extra_row, extra_col] = clearance_value

                else:
                    extra_row = current_row - (max_width + 1) * direction_perp_y
                    extra_col = current_column - (max_width + 1) * direction_perp_x
                    
                    if 0 <= extra_row < rows and 0 <= extra_col < columns:
                        if grid[extra_row, extra_col] == 0:
                            grid[extra_row, extra_col] = clearance_value

    return grid


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1



# TODO +++++++++++++++++++++++++++++++++++++++++++++ TODO
'''
# add option for appending to existing path
'''

def get_paths(template_grid, grid_shape = tuple[int, int], routes: list = None, pads: list = None, 
              draw_flag = False, hide_prints = True, color_matrix = None):
    '''
    Returns grid used during process and paths found as a list(Path)
    Parameters:
    grid                     (array(int)):
    routes                         (list):
    pads                      (list(Pad)):
    width_list                (list(int)):
    hide_prints_flag               (bool):
    '''
    rows, columns = grid_shape
    paths = []
    route_index = 0
    grid = copy.deepcopy(template_grid)
  
    # Initialize Union-Find structure for all points in all routes
    all_points = list(set(point for route in routes for point in route.coord_list))
    point_to_index = {point: idx for idx, point in enumerate(all_points)}
    uf = UnionFind(len(all_points))

    for route in routes:
        netcode = route.netcode
        netname = route.netname
        width   = route.width
        clearance = route.clearance
        points  = route.coord_list
        original_coord = route.original_coord

        grid_copy = copy.copy(grid)
        for pad in pads:
            if pad.original_center in original_coord:
                area = pad.pad_area
                for (y, x) in area:
                    grid_copy[y][x] = 0
        route_index += 1

        if len(points) == 2:
            start_point = points[0]
            dest_point  = points[1]
            path = a_star_search(grid_copy, (rows, columns), start_point, dest_point, netcode, clearance, 
                                    width, False)
            
        else:
            start_point = random.choice(points)
            start_point_idx = point_to_index[start_point]
            available_endpoints = [point for point in points if uf.find(point_to_index[point]) != uf.find(start_point_idx)]
            path, dest_point = lee_search(grid_copy, (rows, columns), start_point, available_endpoints, width, clearance,
                                            netcode, False)   # momentan nu se va putea intersecta;  netcode + 0.7
            
            if dest_point != (-1, -1):
                dest_point_idx = point_to_index[dest_point]
                uf.union(start_point_idx, dest_point_idx)

        if path:
            adjent_path = get_adjent_path(grid = grid_copy, path = path, 
                                          width = width, values = [netcode, netcode + 0.5, netcode + 0.7])
            
            path_found = Path(start_point, dest_point, path, width, clearance, adjent_path)
            paths.append(path_found)

            if hide_prints == False:
                print_path(path = path_found.path, path_index = route_index)

            set_values_in_array(path, grid, route_index)

            extended_path = []
            if adjent_path:
                for subpath in adjent_path:
                    extended_path.extend(subpath)
                    set_values_in_array(subpath, grid, route_index + 0.5)

            grid = mark_clearance_on_grid(grid, (rows, columns), path, width, clearance, netcode + 0.7)
        else:
            print("Unplaced route")
            paths.append(Path(start_point, dest_point, [], width, clearance, []))

    return paths



# check if initial representation is needed and start placing routes
def solution(grid, routes, pads, clearance_list: list = None, width_list: list = None,
              rows = ROWS, columns = COLS, blocked_areas = None, colors = None, draw = False):
    if draw == True:
        color_matrix = get_RGB_matrix(nodes = routes, colors_list = colors, background = COLORS['black'], rows = rows, columns = columns)

        color_matrix = color_pads_in_RGB_matrix(pads = pads, rows = rows, columns = columns, grid = color_matrix, background = COLORS['black'])
        draw_grid(color_matrix = color_matrix, main_path=None)

    clearance_list = [2 for i in range(len(routes))]
    widths = [3 for i in range(len(routes))]
    paths = get_paths(template_grid = grid, grid_shape = (len(grid), len(grid[0])), routes = routes, width_list=widths,
                        pads = pads, clearance_list = clearance_list,
                        color_matrix = color_matrix, draw_flag = draw, 
                        hide_prints = False)
    


# for testing purposes
if __name__ == "__main__":
    rows = ROWS
    columns = COLS
    pins_sizes = 3
    pads = []

    blocked = None

    routes, colors = read_file_routes(file_name='pins.csv', draw = True)

    # for testing Pad class
    for route in routes:
        if len(pads) != 0:
            pin = Pad(center = (route[0], route[1]), original_center = (route[0], route[1]), pad_area = [(route[0], route[1])])
            if pin not in pads:
                pads.append(pin)
            pin = Pad(center = (route[2], route[3]), original_center = (route[2], route[3]), pad_area = [(route[2], route[3])])
            if pin not in pads:
                pads.append(pin)
        else:
            height = 3
            width = 3
            occupied_area = []
            pad = Pad(center = (route[0], route[1]), original_center = (route[0], route[1]), pad_area = [(route[0], route[1])])
            for h in range(height):
                for w in range(width):
                    y, x = pad.center
                    x = x - height // 2 + h 
                    y = y - width // 2 + w
                    occupied_area.append((y, x))
            pad.pad_area = occupied_area
            pads.append(pad)

            occupied_area = []
            pad = Pad(center = (route[2], route[3]), original_center = (route[2], route[3]), pad_area = [(route[2], route[3])])
            for h in range(height):
                for w in range(width):
                    y, x = pad.center
                    y = y - height // 2 + h
                    x = x - width // 2 + w
                    occupied_area.append((y, x))
            pad.pad_area = occupied_area
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
        print(area.pad_area)
    print('\n')
    solution(grid = grid, routes = routes, pads = pads, width_list = widths,
             colors = colors, blocked_areas = blocked, draw = True)







# def get_paths(template_grid, grid_shape = tuple[int, int], routes: list = None, pads: list = None, 
#               existing_paths = None, width_list: list = None, 
#               clearance_list: int = 2, draw_flag = False, hide_prints = True, color_matrix = None):
#     '''
#     Returns grid used during process and paths found as a list(Path)
#     Parameters:
#     grid                     (array(int)):
#     routes                         (list):
#     pads                      (list(Pad)):
#     existing_paths    (list(Path) | None):
#     width_list                (list(int)):
#     color_matrix           (array | None):
#     clearance_list            (list(int)):
#     draw_flag                      (bool):
#     hide_prints_flag               (bool):
#     '''
#     rows, columns = grid_shape
    
#     n = len(routes)
    
#     placed_points = []
#     if existing_paths:
#         paths = copy.deepcopy(existing_paths)
#         for path in existing_paths:
#             placed_points.append([path[0], path[len(path) - 1]])
#     else:
#         paths = []

#     route_index = 0
#     grid = copy.deepcopy(template_grid)

#     for pad in pads:  # nu voi mai avea nevoie de ea deoarece ma voi folosi de matricea template
#         area = pad.pad_area
#         for coord in area:
#             y, x = coord
#             grid[y][x] = -1

#     for route in routes:
#         route_index += 1
#         st_row, st_col, dest_row, dest_col = route[0:4]

#         # 0 = unblock; -1  = blocked; 1,2,... = used
#         grid_copy = copy.copy(grid)

#         for pad in pads:
#             y, x = pad.center
#             if (st_row, st_col) == (y, x) or (dest_row, dest_col) == (y, x):
#                 area = pad.pad_area
#                 for coord in area:
#                     y, x = coord
#                     grid_copy[y][x] = 0         
        
#         if not choose_a_star():
#             path = lee_search(grid = grid_copy, grid_size = (rows, columns), route_values = [route_index, route_index + 0.5, route_index + 0.7], 
#                                 start = (st_row, st_col), possible_ends = [(dest_row, dest_col)],
#                                 width = width_list[route_index-1], clearance = clearance_list[route_index-1], hide_prints = False)
                
#         else:
#             path = a_star_search(grid = grid_copy, grid_size = (rows, columns), path_value = route_index,
#                                     start = (st_row, st_col), destination = (dest_row, dest_col),
#                                     width = width_list[route_index-1], clearance = clearance_list[route_index-1],
#                                     hide_prints = hide_prints)
          
#         if path:
#             adjent_path = get_adjent_path(grid = grid_copy, path = path, 
#                                           width = width_list[route_index-1], values = [0, route_index])

#             path_found = Path(start = (st_row, st_col), destination = (dest_row, dest_col), 
#                               width = width_list[route_index-1], clearance = clearance_list[route_index-1],
#                               path = path, other_nodes = adjent_path)

#             if hide_prints == False:
#                 print_path(path = path_found.path, path_index = route_index)
            
#             paths.append(path_found)

#             set_values_in_array(path, grid, route_index)

#             extended_path = []
#             if adjent_path:
#                 for subpath in adjent_path:
#                     extended_path.extend(subpath)
#                     set_values_in_array(subpath, grid, route_index + 0.5)

#             grid = mark_clearance_on_grid(grid, (rows, columns), path, width_list[route_index-1], 
#                                           clearance_list[route_index-1], route_index + 0.7)

#             if draw_flag == True:    
#                 draw_grid(color_matrix = color_matrix, main_path = path,
#                             color_main_path = random.choice([COLORS['yellow'], COLORS['red'], COLORS['pink'], COLORS['orange']]),
#                             other_nodes = extended_path, color_other_nodes = random.choice([COLORS['yellow'], COLORS['red'], COLORS['pink'], COLORS['orange']])) 
            
#         else:
#             if hide_prints == False:
#                 print("\tNo change in drawing. Route can't be placed\n")
#             paths.append(Path(start = (st_row, st_col), destination = (dest_row, dest_col), 
#                               width = width_list[route_index-1], clearance=clearance_list[route_index-1],
#                               path = [], other_nodes = []))
#     return paths
import numpy as np # --
import heapq
from collections import deque
import copy
import random
from utils import Cell, Path, is_destination, is_unblocked, is_valid, \
                    h_euclidian, get_perpendicular_direction, check_90_deg_bend


# equivalent to mark_apth_in)grid from pcb_utils
def set_values_in_array(blocked_cells, arr, value: 0): 
    # value = 1, 2, ... for routes, 0 for other types of obstacle
    arr_copy = copy.deepcopy(arr)
    if blocked_cells:
        for cell in blocked_cells:
            i, j = cell
            arr_copy[i][j] = value
    return arr_copy


def get_YX_directions(current_poz: tuple[int, int], previous_poz: tuple[int, int]):
    '''Returns distance for X and Y between 2 points reprezented as (int, int)'''
    return current_poz[0] - previous_poz[0], current_poz[1] - previous_poz[1]


def get_nodes(grid, grid_shape, point: tuple[int, int], width: int, direction_y: int, direction_x: int, values: int):
    '''
    Returns a list for cell neighbors according to width of the path
    '''
    nodes = []
    rows, columns = grid_shape
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


def get_adjent_path(grid, grid_shape, path, width: int, values):
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
        direction_y, direction_x = get_YX_directions(current_poz = (current_row, current_column), 
                                                            previous_poz = (previous_row, previous_column))
        direction_perp_y, direction_perp_x = get_perpendicular_direction(direction_y, direction_x) # direction perpendicular to x and y
        neighbors = get_nodes(grid = grid, grid_shape = grid_shape, point = (current_row, current_column), width = width,
                               direction_y = direction_perp_y, direction_x = direction_perp_x, values = values)
        other_nodes.append(neighbors)
    
    return other_nodes


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



# A star algorihm (modified Dijkstra)
def a_star_search(grid, grid_size: tuple[int, int], start: tuple[int, int], destination: tuple[int, int], 
                  path_value: int, clearance: int = 1, width: int = 1, hide_prints = True):
    rows, columns = grid_size
    start_row, start_col = start
    destination_row, destiantion_col = destination
    
    if not is_valid((start_row, start_col), (rows, columns)) or not is_valid((destination_row, destiantion_col), (rows, columns)):
        if hide_prints == False:
            print("\nStart | Dest invalid")
        return False
    
    if not is_unblocked(grid, (start_row, start_col), values = [0, path_value]) or not \
        is_unblocked(grid, (destination_row, destiantion_col), values = [0, path_value]):   # de testat 0 sau 0, netcode, netcode + 0.5, netcode + 0.7
        if hide_prints == False:
            print("\nStart | Dest blocked")
        return False

    # Return the path from source to destination
    def get_path():
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
        #print("Progress:", i, j, ":", start_row, start_col, ":", destination_row, destiantion_col)

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
                        aux_direction = cell_details[aux_y][aux_x].direction
                        if aux_direction and check_90_deg_bend(aux_direction, parent_direction):
                            nr_90_degree_bends += 1                                    

                    g_new = parent_cell.g + h_euclidian((i, j), (new_i, new_j))   # greedy aprouch
                    h_new = h_euclidian((new_i, new_j), (destination_row, destiantion_col)) * 7
                    f_new = g_new + h_new + (nr_90_degree_bends << 15) + (nr_bends << 10)

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

                    path = get_path()
                    return path

    if hide_prints == False:
        print("\nDestination not reached")
    return []



# routing with lee
def lee_search(grid, grid_size: tuple, start: tuple[int, int], possible_ends: list,
               width: int, clearance: int, value: int, hide_prints: bool = True):
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
    visited = set()

    # Add start to BFS q
    i, j = start_row, start_column
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
            # elif grid[i][j] == value:
            #     end_reached = True
            #     if not hide_prints:
            #         print(f"\n\nDestination (route) ({i}, {j}) reached")

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
                 (new_i, new_j) not in visited: #visited[new_i][new_j] == False:

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


'''
Add option for appending to existing path --- TODO
'''

def get_paths(template_grid, grid_shape: tuple[int, int], planned_routes: dict, routing_order: list, 
              pads: list = None, hide_prints = True):
    rows, columns = grid_shape
    paths = []
    route_index = 0
    grid = copy.deepcopy(template_grid)
  
    # Initialize Union-Find structure for all points in all routes
    # all_points = list(set(point for route in routes for point in route.coord_list))
    # # de adaugat eliminarea componentelor / rutelor daca exista drum de la x la y
    # point_to_index = {point: idx for idx, point in enumerate(all_points)}
    # uf = UnionFind(len(all_points))

    all_points = list(set(point for netcode in routing_order for point in planned_routes[netcode].coord_list))
    point_to_index = {point: idx for idx, point in enumerate(all_points)}
    uf = UnionFind(len(all_points))

    for netcode in routing_order:
        route = planned_routes[netcode]
        # netcode = route.netcode
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
            adjent_path = get_adjent_path(grid = grid_copy, grid_shape = (rows, columns), path = path, 
                                          width = width, values = [netcode, netcode + 0.5, netcode + 0.7])
            
            path_found = Path(start_point, dest_point, netcode, path, width, clearance, adjent_path)
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
            paths.append(Path(start_point, dest_point, netcode, [], width, clearance, []))

    return paths
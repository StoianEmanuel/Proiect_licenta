# stores functions related to A_star.py and GA_routing
from math import sqrt
import copy
import re


# class used for A* search, that stores 2 types of costs: so far and remaining
class Cell:
    '''Class used for A star search to determine the cost of paths based on heuristic\n
    Attributes:
        parent_y (int): Y coord of Cell's parent
        parent_x (int): X coord of Cell's parent (from where is accessed)
        f (float): Total cost; f = g + h
        h (float): Cost from starting cell
        g (float): Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest
    '''
    def __init__(self, parent = None, f = float('inf'), h = float('inf'), g = 0, 
                 direction = None, nr_bends: int = 0, nr_90_deg_bends: int = 0):
        if parent:
            self.parent = parent
        else:
            self.parent = (-1, -1)
        self.f = f  # Total cost (h + g)
        self.h = h  # Cost from start to cell
        self.g = g  # Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest
        self.direction = direction
        self.bends = nr_bends
        self.bends_90_deg = nr_90_deg_bends



# class used to define a path between two points that has width = n x 1;  
class Path:
    """Class used to define a path between two points with a specified width.\n
    Attributes:
        start_x      (int): The x-coordinate of the starting point.
        start_y      (int): The y-coordinate of the starting point.
        dest_x       (int): The x-coordinate of the destination point.
        dest_y       (int): The y-coordinate of the destination point.
        path        (list): The list of points representing the main path.
        width        (int): The width of the path.
        other_nodes (list): Additional points related to the main path.
    """
    def __init__(self, start: tuple[int, int], destination: tuple[int,  int], netcode: int = 0,
                 path = None, width: int = 1, clearance: int = 1, mutated: bool = False, simplified_path = None, layer_id = 0):
        self.start          = start
        self.destination    = destination
        self.netcode        = netcode
        self.path           = path
        self.simplified_path= simplified_path if simplified_path else []
        self.width          = width
        self.mutated        = mutated
        self.clearance      = clearance
        self.layer_id       = layer_id

    def update_simplified_path(self):
        self.simplified_path = simplify_path(self.path)


class Pad:
    '''
    Class used to define a pad for a part.
    Attributes:
        center_x       (int, int): (Y, X) coord of pad's center  
        pad_area           (list): (Y, X) coord occupied by pad on board; useful for irregular shapes or circles
    '''
    def __init__(self, center: tuple[int, int], original_center: tuple[int, int], 
                 pad_area = None, netcode = None, netname = None, clearance = None, clearance_area = None):
        self.center         = center
        self.original_center= original_center  # inainte de transformari (in nm)
        self.pad_area       = pad_area   # coord care realizeaza poligonul - lista de tupluri
        self.netcode        = netcode    # folosit pentru a determina carui net ii sunt asociate
        self.netname        = netname    # posibil folosit pentru a crea folosi custom clearance si width
        self.clearance      = clearance
        self.clearance_area = clearance_area


class PlannedRoute:
    def __init__(self, netcode, netname, width: int, clearance: int, coord_list, original_coord = None, existing_conn = None, layer_id = 0):
        self.netcode = netcode
        self.netname = netname
        self.width = width
        self.clearance = clearance
        self.coord_list = coord_list
        self.original_coord = original_coord if original_coord else []
        self.existing_conn = existing_conn if existing_conn else []
        self.layer_id = layer_id # True => Top, False => Bottom
    
    def add_track(self, track):
        self.coord_list.append(track)
    
    def add_existing_conn(self, conn):
        self.existing_conn.append(conn)

    def set_netname(self, netname):
        self.netname = netname


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


# User settings
class UserSettings:
    def __init__(self):
        self.dict = {
            'POW': {'clearance': 500000, 'width': 1000000, 'enabled': False, 'layer_id': 'B.Cu',
                    'pattern': re.compile(r'POW|POWER|\+\d+V|-\d+V|VDD|VCC')},
            'GND': {'clearance': 200000, 'width': 500000, 'enabled': False, 'layer_id': 'B.Cu',
                    'pattern': re.compile(r'GND|0V|VSS|VEE')},
            'ALL': {'clearance': 200000, 'width': 500000, 'enabled': True, 'layer_id': 'B.Cu',
                    'pattern': re.compile(r'.*')}
        }
        self.keep = False
        self.layers = 1

    def change_to_multiple_layers(self):
        if self.layers == 2:
            self.dict['POW']['layer'] = 'F.Cu'

    def change_settings(self, factor):
        for key, values in self.dict.items():
            clearance = division_int(values['clearance'], factor)
            width = division_int(values['width'], factor) 
            self.dict[key]['width'] = width
            self.dict[key]['clearance'] = clearance

    def set_width(self, key, width):
        if key in self.dict:
            self.dict[key]['width'] = width


    def set_clearance(self, key, width):
        if key in self.dict:
            self.dict[key]['width'] = width

 
    def get_multiplier(self):
        value = 100000
        for values in self.dict.values():
            if values['enabled']:
                clearance = values['clearance']
                width = values['width']
                if clearance % 10000 !=0 or width % 10000 != 0:
                    value = 1000
                elif clearance % 100000 !=0 or width % 100000 != 0:
                    value = 10000
        return value


def division_int(value, factor):
    return value // factor


# check if cell / move is valid
def is_unblocked(array, point: tuple[int, int], values: list):
    row, column = point
    return array[row][column] in values


# check if cell is inside the grid
def is_valid(point: tuple[int, int], array_shape: tuple[int, int]):
    row, column = point
    total_rows, total_columns = array_shape
    return 0 <= row < total_rows and 0 <= column < total_columns


# check if dest is reached
def is_destination(current_point: tuple[int, int], destination_point: tuple[int, int]):
    return current_point == destination_point

# poate voi insera si partea de bends
def get_route_length(route):        # route = [[,,,] - start,   ..., [,,,], ... ,      [,,,] - dest]
    distance = 0
    n = len(route)-1
    for index in range(n):
        p1_row, p1_column = route[index]
        p2_row, p2_column = route[index+1]
        distance += h_euclidian((p1_row, p1_column), (p2_row, p2_column))
    return distance


def check_90_deg_bend(d1: tuple[int, int], d2: tuple[int, int]):
    return d1[0] * d2[1] - d2[0] * d1[1] == 0


def get_number_of_bends(path):
    if len(path) < 3:
        return 0, 0
    
    nr_regular_bends = 0
    nr_90_bends = 0

    for index in range(1, len(path)-1):
        dy1, dx1 = path[index][0] - path[index-1][0], path[index][0] - path[index-1][0]
        dy2, dx2 = path[index][0] - path[index+1][0], path[index][0] - path[index+1][0]
        if (dy1, dx1) != (dy2, dx2):
            nr_regular_bends += 1
            if check_90_deg_bend((dy1, dx1), (dy2, dx2)): 
                nr_90_bends = nr_90_bends + 1 

    return nr_regular_bends, nr_90_bends 


def fitness_function(paths, unplaced_routes_number: int, unplaced_route_penalty = 2):
    total_length = 0
    total_regular_bends, total_90_bends = 0, 0
    for path in paths:
        route_len = get_route_length(path)
        nr_regular_bends, nr_90_bends = get_number_of_bends(path)

        total_length += route_len
        total_regular_bends += nr_regular_bends
        total_90_bends += nr_90_bends

    total_length = total_length * (unplaced_route_penalty ** unplaced_routes_number) + total_regular_bends + (total_90_bends << 4)

    return total_length


# function that save for each path only the points (x, y) that are start, destionation or represents a intersection between 2 lines
# forms an angle 
def simplify_path(path):
    simplified_path = []      # most significant points - start, stop, "bend" points
    if path:
        length = len(path)
        simplified_path.append(path[0])
        for i in range(1, length-1):
            current_point = path[i]
            prev_point = path[i - 1]
            next_point = path[i + 1]

            # Calculate direction vectors
            direction_current = (next_point[0] - current_point[0], next_point[1] - current_point[1])
            direction_previous = (current_point[0] - prev_point[0], current_point[1] - prev_point[1])

            # Check if the direction has changed
            if direction_current != direction_previous:
                simplified_path.append(current_point)

        simplified_path.append(path[-1])
    return simplified_path


def get_simplified_paths(paths_list):
    simplified_paths = []
    for path in paths_list:
        p = simplify_path(path)
        simplified_paths.append(p)
    return simplified_paths


# Returns tuple (dir_y_perp, dir_x_perp) so segment [(P.y, P.x), (P.y + dir_y, P.x + dir_x)] 
# is perpendicular to [(P.y, P.x), (P.y + dir_perp_y, P.x + dir_perp_x)]
def get_perpendicular_direction(direction_y: int, direction_x: int):
    if direction_y == 0:  # orizontal
        direction_perp_y = 1
        direction_perp_x = 0
    elif direction_x == 0:
        direction_perp_y = 0
        direction_perp_x = 1
    elif abs(direction_y) == abs(direction_x):
        direction_perp_y = -direction_x
        direction_perp_x = direction_y
    else:   # abs(dir_x) != abs(dir_y) --- (-1,1), (1,-1)
        direction_perp_y = direction_y
        direction_perp_x = -direction_x

    return direction_perp_y, direction_perp_x


'''movement heuristics types'''
# 4 directions
def h_manhattan(point1: tuple[int, int], point2: tuple[int, int]):
    return abs(point1[1] - point2[1]) + abs(point1[0] - point2[0])

# any direction
def h_euclidian(point1: tuple[int, int], point2: tuple[int, int]):
    return sqrt((point1[1] - point2[1])**2 + (point1[0] - point2[0])**2)
''''''


def get_YX_directions(current_poz: tuple[int, int], previous_poz: tuple[int, int]):
    '''Returns distance for X and Y between 2 points reprezented as (int, int)'''
    return current_poz[0] - previous_poz[0], current_poz[1] - previous_poz[1]



def mark_path_in_array(array, path, value, overwrite = True):
    ''' value = value assigned to route'''
    aux = copy.deepcopy(array)
    try:
        for vertex in path:
            y, x = vertex
            if overwrite or aux[y][x] == 0:
                aux[y][x] = value
    except Exception as e:
        print("Error ", e, "while marking path in array")
    return aux



def update_grid_with_paths(grid, grid_shape: tuple[int, int], previous_paths):
    rows, column = grid_shape
    for prev_path in previous_paths:
        netcode = prev_path.netcode
        path = prev_path.path
        width = prev_path.width
        clearance = prev_path.clearance
        layer = prev_path.layer_id
        grid[layer][:][:] = mark_path_in_array(grid[layer][:][:], path, netcode)
        grid[layer][:][:] = mark_adjent_path(grid[layer][:][:], (rows, column), path, width, netcode)
        grid[layer][:][:] = mark_clearance_on_grid(grid[layer][:][:], (rows, column), path, width, clearance, netcode)
    return grid


def mark_adjent_path(grid, grid_shape, path, width: int, netcode: int):
    #values = [0, netcode, netcode + 0.5, netcode + 0.7]
    value = netcode + 0.5
    overwrite_values = [0, netcode + 0.7]
    rows, columns = grid_shape
    side = (width - 1) // 2
    prev_row, prev_column = path[0]
    if path:
        for index in range(1, len(path)-1):
            row, column = path[index]
            direction_y, direction_x = get_YX_directions((row, column), (prev_row, prev_column))
            for i in range(1, side + 1):
                new_row = row + i * direction_y
                new_col = column + i * direction_x
                if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), overwrite_values):
                    grid[new_row][new_col] = value

                new_row = row - i * direction_y
                new_col = column - i * direction_x
                if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), overwrite_values):
                    grid[new_row][new_col] = value
            
            if width % 2 == 0: # asymetric case; side widths: n, n+1
                new_row = row + (side + 1) * direction_y
                new_col = column + (side + 1) * direction_x
                if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), overwrite_values):
                    if new_row > row or (new_row == row and new_col == column):
                        grid[new_row][new_col] = value
                else:
                    new_row = row - (side + 1) * direction_y
                    new_col = column - (side + 1) * direction_x # 2
                    if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), overwrite_values):
                        if new_row > row or (new_row == row and new_col == column):
                            grid[new_row][new_col] = value

            prev_row, prev_column = row, column

    return grid



def mark_clearance_on_grid(grid, grid_shape, path, path_width: int, clearance_width: int, netcode: int):
    """
    Marks the grid with clearance values around the given path.

    Parameters:
    grid            (array): The grid to be marked.
    path            (list) : List of (X, Y) coordinates for the main path.
    clearance_width (int)  : The width of the clearance to be added around the path.
    path_width      (int)  : The width of the path.
    clearance_value (int)  : The value to mark for clearance on the grid.
    """
    clearance_value = netcode + 0.7
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
from math import sqrt
import copy
import re


class Cell:
    '''Class used for A star search to determine the cost of paths based on heuristic'''
    def __init__(self, parent = None, f = float('inf'), h = float('inf'), g = 0):
        self.parent = parent if parent else (0, 0)
        self.f = f  # Total cost (h + g)
        self.h = h  # Cost from start to cell
        self.g = g  # Heuristic (Manhattan / Euclidian / Diagonal) cost from current cell to dest


    def update(self, f, h, g, parent):
        self.f = f
        self.h = h
        self.g = g
        self.parent = parent


class Path:
    """Class used to define a path between two points with a specified width"""
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
    '''
    def __init__(self, center: tuple[int, int], original_center: tuple[int, int], 
                 netcode = None, clearance_area = None):
        self.center         = center
        self.original_center= original_center  # inainte de transformari (in nm)
        self.netcode        = netcode    # folosit pentru a determina carui net ii sunt asociate
        self.clearance_area = clearance_area


class PlannedRoute:
    '''Class used to define a route found for a netclass'''
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
        self.size = [1] * n  # Inițializăm dimensiunea fiecărei componente cu 1
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if root_u > root_v:
                self.parent[root_u] = root_v
                self.size[root_v] += self.size[root_u]
                self._update_all_descendants(root_u, root_v)
            else:
                self.parent[root_v] = root_u
                self.size[root_u] += self.size[root_v]
                self._update_all_descendants(root_v, root_u)
    
    def _update_all_descendants(self, old_root, new_root):
        for i in range(len(self.parent)):
            if self.find(i) == old_root:
                self.parent[i] = new_root


# User settings
class UserSettings:
    def __init__(self):
        self.dict = {
            'POW': {'clearance': 500000, 'width': 1000000, 'enabled': False, 'layer_id': 'B.Cu',
                    'pattern': re.compile(r'POW|POWER|\+\d+V|-\d+V|VDD|VCC|U\d+\-\+')},
            'GND': {'clearance': 200000, 'width': 500000, 'enabled': False, 'layer_id': 'B.Cu',
                    'pattern': re.compile(r'GND|0V|VSS|VEE|U\d+\-\-')},
            'ALL': {'clearance': 200000, 'width': 500000, 'enabled': True, 'layer_id': 'B.Cu',
                    'pattern': re.compile(r'.*')}
        }
        self.keep = False
        self.layers = 1


    def change_settings(self, factor):
        for key, values in self.dict.items():
            clearance = division_int(values['clearance'], factor)
            width = division_int(values['width'], factor) 
            self.dict[key]['width'] = width
            self.dict[key]['clearance'] = clearance


    def set_layer(self, key, layer):
        if key in self.dict:
            self.dict[key]['layer_id'] = layer


    def set_width(self, key, width):
        if key in self.dict:
            self.dict[key]['width'] = width


    def set_clearance(self, key, clearance):
        if key in self.dict:
            self.dict[key]['clearance'] = clearance

 
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


class Individual:
        '''Describes a possible solution for GA routing problem'''
        def __init__(self):
            self.order = None
            self.paths = None
            self.unplaced_routes_number = 100
            self.total_cost = -1
        
        def set_values(self, order = [], paths: Path = None, unplaced_routes_number: int = 0, paths_total_cost = -1):
            self.order  = order    # holds order of routes from input
            self.paths  = paths    # [[path1], [path2], ...]
            self.unplaced_routes_number = unplaced_routes_number # from the necessary ones
            self.total_cost = paths_total_cost      # fitness_value

        def __repr__(self):
            return repr((self.order, self.paths, self.unplaced_routes_number, self.total_cost))
        
        def get_path_cost(self):
            return self.total_cost

        def __str__(self) -> str:
            return f'Order: {self.order}; Total cost: {self.total_cost}'


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


def check_bend(parent_coord, current_coord, new_coord):
    # Determină direcția din parent către current
    if parent_coord == current_coord:
        return False
    dir1 = (current_coord[0] - parent_coord[0], current_coord[1] - parent_coord[1])
    # Determină direcția din current către new
    dir2 = (new_coord[0] - current_coord[0], new_coord[1] - current_coord[1])
       
    return dir1 != dir2


def check_90_deg_bend(p1, p2, p3):
    y1, x1 = p1
    y2, x2 = p2
    y3, x3 = p3
    
    # Calculate the sides
    A = (x2 - x1)**2 + (y2 - y1)**2
    B = (x3 - x2)**2 + (y3 - y2)**2
    C = (x3 - x1)**2 + (y3 - y1)**2
     
    # Check Pythagoras Formula 
    if A == (B + C) or B == (A + C) or C == (A + B):
        return True
    return False


# Returns no of 90 deg bends, and total no of bends for a specific list of coordinates 
def get_number_of_bends(path):
    if len(path) < 3:
        return 0, 0
    
    nr_regular_bends = 0
    nr_90_bends = 0
    p1 = path[0]
    p2 = path[1]
    p3 = path[2]
    for index in range(1, len(path)-2):
        if check_bend(p1, p2, p3):
            nr_regular_bends += 1
            if check_90_deg_bend(p1, p2, p3): 
                nr_90_bends += 1
        p1 = p2
        p2 = p3
        p3 = path[index+2]

    return nr_regular_bends, nr_90_bends

# Associate score to individuals based on tracks total length, unplaced nets and number of bends
def fitness_function(paths, unplaced_routes_number: int, unplaced_route_penalty = 2):
    total_length = 0
    total_regular_bends, total_90_bends = 0, 0
    for path in paths:
        if path:
            route_len = get_route_length(path)
            nr_regular_bends, nr_90_bends = get_number_of_bends(path)

            total_length += route_len
            total_regular_bends += nr_regular_bends
            total_90_bends += nr_90_bends

    total_length = round((total_length + total_regular_bends + (total_90_bends << 5)) * (unplaced_route_penalty << unplaced_routes_number), 7)

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

# Returns a list with tracks
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
                log_modificari("mark_path", array, "grid[layer_id]", y, x, value)
                aux[y][x] = value
                
    except Exception as e:
        print("Error ", e, "while marking path in array")
    return aux



def update_grid_with_paths(array, grid_shape: tuple[int, int], previous_paths):
    grid = copy.deepcopy(array)
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


def log_modificari(nume_functie, grid, object, row, column,value):
    with open("log_modificari.txt", "a") as file:
        file.write(f"Functia {nume_functie} a modificat celula obiectului {object} : {(row, column)} din {grid[row, column]} in valoarea {value}\n")




def mark_adjent_path(grid, grid_shape, path, width: int, netcode: int):
    #values = [0, netcode, netcode + 0.5, netcode + 0.7]
    value = netcode + 0.5
    overwrite_values = [0, netcode + 0.7]
    rows, columns = grid_shape
    side = (width - 1) // 2

    if path:
        prev_row, prev_column = path[1]
        for index in range(2, len(path)-2):
            row, column = path[index]
            direction_y, direction_x = get_YX_directions((row, column), (prev_row, prev_column))
            dir_perp_y, dir_perp_x = get_perpendicular_direction(direction_y, direction_x)
            for i in range(1, side+1):
                new_row = row + i * dir_perp_y
                new_col = column + i * dir_perp_x
                if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), overwrite_values):
                    log_modificari("adjent", grid, "grid[layer_id]", new_row, new_col, value)
                    grid[new_row][new_col] = value

                new_row = row - i * dir_perp_y
                new_col = column - i * dir_perp_x
                if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), overwrite_values):
                    log_modificari("adjent", grid, "grid[layer_id]", new_row, new_col, value)
                    grid[new_row][new_col] = value
            
            if not width % 2: # asymetric case; side widths: n, n+1
                new_row = row + (side + 1) * dir_perp_y
                new_col = column + (side + 1) * dir_perp_x
                if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), overwrite_values):
                    if new_row > row or (new_row == row and new_col == column):
                        log_modificari("adjent", grid, "grid[layer_id]", new_row, new_col, value)
                        grid[new_row][new_col] = value
                else:
                    new_row = row - (side + 1) * dir_perp_y
                    new_col = column - (side + 1) * dir_perp_x # 2
                    if is_valid((new_row, new_col), (rows, columns)) and is_unblocked(grid, (new_row, new_col), overwrite_values):
                        if new_row > row or (new_row == row and new_col == column):
                            log_modificari("adjent", grid, "grid[layer_id]", new_row, new_col, value)
                            grid[new_row][new_col] = value

            prev_row, prev_column = row, column

    return grid



def mark_clearance_on_grid(grid, grid_shape, path, path_width, clearance_width, netcode):
    clearance_value = netcode + 0.7
    rows, columns = grid_shape
    half_width = (path_width - 1) // 2
    max_width = half_width + clearance_width + 1
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0),  (1, 1)]

    for i in range(2, len(path)-2):
        current_row, current_column = path[i]
        next_row, next_column = path[i + 1]
        
        direction_y, direction_x = next_row - current_row, next_column - current_column
        direction_perp_y, direction_perp_x = get_perpendicular_direction(direction_y, direction_x)

        for j in range(half_width + 1, max_width):
            new_row = current_row + j * direction_perp_y
            new_col = current_column + j * direction_perp_x
            for t, z in directions:
                neighbor_row = new_row + t
                neighbor_column = new_col + z  
                if is_valid((neighbor_row, neighbor_column), (rows, columns)) and is_unblocked(grid, (neighbor_row, neighbor_column), [0]):
                    log_modificari("clr", grid, "grid[layer_id]", neighbor_row, neighbor_column, clearance_value)
                    grid[neighbor_row, neighbor_column] = clearance_value

            new_row = current_row - j * direction_perp_y
            new_col = current_column - j * direction_perp_x
            for t, z in directions:
                neighbor_row = new_row + t
                neighbor_column = new_col + z  
                if is_valid((neighbor_row, neighbor_column), (rows, columns)) and is_unblocked(grid, (neighbor_row, neighbor_column), [0]):
                    log_modificari("clr", grid, "grid[layer_id]", neighbor_row, neighbor_column, clearance_value)
                    grid[neighbor_row, neighbor_column] = clearance_value

        # Handle the asymmetric case if path_width is even
        if not path_width % 2:
            extra_row = current_row + (max_width + 1) * direction_perp_y
            extra_col = current_column + (max_width + 1) * direction_perp_x
            
            for t, z in directions:
                neighbor_row = extra_row + t
                neighbor_column = extra_col + z  
                if is_valid((neighbor_row, neighbor_column), (rows, columns)) and is_unblocked(grid, (neighbor_row, neighbor_column), [0]):
                    log_modificari("clr", grid, "grid[layer_id]", neighbor_row, neighbor_column, clearance_value)
                    grid[neighbor_row, neighbor_column] = clearance_value

                else:
                    extra_row = current_row - (max_width + 1) * direction_perp_y
                    extra_col = current_column - (max_width + 1) * direction_perp_x
                    
                    for t, z in directions:
                        neighbor_row = extra_row + t
                        neighbor_column = extra_col + z  
                        if is_valid((neighbor_row, neighbor_column), (rows, columns)) and is_unblocked(grid, (neighbor_row, neighbor_column), [0]):
                            log_modificari("clr", grid, "grid[layer_id]", neighbor_row, neighbor_column, clearance_value)
                            grid[neighbor_row, neighbor_column] = clearance_value
    
    return grid

# Returns a list that contains unique elements in order
def list_unique(seq, id_f=None): 
   if id_f is None:
       def id_f(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = id_f(item)
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result
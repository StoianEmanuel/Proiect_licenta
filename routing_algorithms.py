import numpy as np # --
import heapq
from collections import deque
import copy
from utils import Cell, Path, UnionFind, is_destination, is_unblocked, is_valid, \
                    h_euclidian, get_perpendicular_direction, check_90_deg_bend, \
                    mark_path_in_array, mark_adjent_path, mark_clearance_on_grid, check_bend

# Check is cell is unlocked relative to asociated netcode
def is_valid_and_unlocked(array, neighbor_row, neighbor_column, rows, columns, values):
    valid = is_valid((neighbor_row, neighbor_column), (rows, columns))
    if valid:
        unblocked = is_unblocked(array, (neighbor_row, neighbor_column), values)
        return False if not unblocked else array[neighbor_row][neighbor_column] % 10 < 0.6
    return True


# Check current cell's neighbors for cleareance constraint
def check_3x3_square(array, point: tuple[int, int], array_shape: tuple[int, int], values):
    rows, columns = array_shape
    current_row, current_column = point

    neighbors = [(current_row - 1, current_column - 1),
                 (current_row - 1, current_column    ),
                 (current_row - 1, current_column + 1),
                 (current_row    , current_column - 1),
                 (current_row    , current_column    ),
                 (current_row    , current_column + 1),
                 (current_row + 1, current_column - 1),
                 (current_row + 1, current_column    ),
                 (current_row + 1, current_column + 1),]
    
    # loop unrolling for cells verfications instead of directions = [(-1,-1), (-1,0), ... , (1,1)]
    if not is_valid_and_unlocked(array, neighbors[0][0], neighbors[0][1], rows, columns, values):
        return False
    if not is_valid_and_unlocked(array, neighbors[1][0], neighbors[1][1], rows, columns, values):
        return False
    if not is_valid_and_unlocked(array, neighbors[2][0], neighbors[2][1], rows, columns, values):
        return False
    if not is_valid_and_unlocked(array, neighbors[3][0], neighbors[3][1], rows, columns, values):
        return False
    if not is_valid_and_unlocked(array, neighbors[4][0], neighbors[4][1], rows, columns, values):
        return False
    if not is_valid_and_unlocked(array, neighbors[5][0], neighbors[5][1], rows, columns, values):
        return False
    if not is_valid_and_unlocked(array, neighbors[6][0], neighbors[6][1], rows, columns, values):
        return False
    if not is_valid_and_unlocked(array, neighbors[7][0], neighbors[7][1], rows, columns, values):
        return False  
    
    return True


# Check row (vert, horiz, diag) assigned to current cell for clearance constraint
def check_line(array, point: tuple[int, int], sign_y: int, sign_x: int, 
               array_shape: tuple[int, int], offset: int, values):
    current_row, current_column = point
    rows, columns = array_shape
    for i in range(1, offset + 1):
        new_row = current_row + i * sign_y
        new_col = current_column + i * sign_x
        if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), values):
                return False
    return True



def check_width_and_clearance(array, array_shape: tuple[int, int], point: tuple[int, int], direction_y: int, direction_x: int, 
                              path_values: int, width: int = 1, clearance: int = 1):
    rows, columns = array_shape
    row, column = point
    half_width = (width - 1) // 2
    direction_perp_y, direction_perp_x = get_perpendicular_direction(direction_y, direction_x)
    values = path_values
    even_width = False if width % 2 else True
    total_width = half_width + clearance
    def check_main_lines():
        return check_line(array, (row+direction_y, column+direction_x), direction_perp_y, direction_perp_x, (rows, columns), half_width, values) \
                and check_line(array, (row+direction_y, column+direction_x), -direction_perp_y, -direction_perp_x, (rows, columns), half_width, values)

    def check_clearance_lines():
        for i in range(half_width, total_width):
            if not check_3x3_square(array, (row + direction_y + i * direction_perp_y,
                                            column + direction_x + i * direction_perp_x), (rows, columns), values) or \
                not check_3x3_square(array, (row + direction_y - i * direction_perp_y,
                                            column + direction_x - i * direction_perp_x), (rows, columns), values):
                    return False
        if even_width:
            if not check_3x3_square(array, (row + direction_y + total_width * direction_perp_y,
                                                column + direction_x + total_width * direction_perp_x), (rows, columns), values):
                if not check_3x3_square(array, (row + direction_y - total_width * direction_perp_y,
                                                    column + direction_x - total_width * direction_perp_x), (rows, columns), values):
                    return False
        return True

    def check_even_width_asymmetry():
        if even_width:
            new_row = row + direction_y + (half_width + 1) * direction_perp_y
            new_col = column + direction_x + (half_width + 1) * direction_perp_x
            if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), values) or \
                not check_3x3_square(array, (new_row + clearance * direction_perp_y, new_col + clearance * direction_perp_x),
                                  (rows, columns), values):
                new_row = row + direction_y - (half_width + 1) * direction_perp_y
                new_col = column + direction_x - (half_width + 1) * direction_perp_x
                if not is_valid((new_row, new_col), (rows, columns)) or not is_unblocked(array, (new_row, new_col), values) or \
                    not check_3x3_square(array, (new_row - clearance * direction_perp_y, new_col - clearance * direction_perp_x),
                                      (rows, columns), values):
                    return False
        return True

    if not check_main_lines():
        return False
    if not check_even_width_asymmetry():
        return False
    if not check_clearance_lines():
        return False

    return True


# A star algorihm (modified Dijkstra)
def a_star_search(grid, grid_size: tuple[int, int], start: tuple[int, int], destination: tuple[int, int], 
                  netcode: int, clearance: int = 1, width: int = 1):
    rows, columns = grid_size
    start_row, start_col = start
    destination_row, destiantion_col = destination

    if not is_valid((start_row, start_col), (rows, columns)) or not is_valid((destination_row, destiantion_col), (rows, columns)):
        return []
    
    if not is_unblocked(grid, (start_row, start_col), values = [0, netcode]) or not \
        is_unblocked(grid, (destination_row, destiantion_col), values = [0, netcode]):   # de testat 0 sau 0, netcode, netcode + 0.5, netcode + 0.7
        return []

    # Return the path from source to destination
    def get_path():
        path = []
        row, column = destination
        
        while not cell_details[row][column].parent == (row, column):
            path.append((row, column))
            row, column = cell_details[row][column].parent
        
        path.append((row, column)) # add start node to path
        path.reverse()
        return path

    # initialize start of the list
    i = start_row
    j = start_col
    
    cell_details = [[Cell() for _ in range(columns)] for _ in range(rows)] # status of every cell in the grid
    cell_details[i][j].update(f = 0, g = 0, h = 0, parent = (i,j))

    open_list = []  # cells to be visited
    heapq.heappush(open_list, (0.0, i, j))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
  
    while len(open_list) > 0:
        _, i, j = heapq.heappop(open_list)

        for dir in directions:  # for each direction check the succesor
            dir_y, dir_x = dir
            new_i = i + dir_y
            new_j = j + dir_x

            if is_valid((new_i, new_j), (rows, columns)) and \
                is_unblocked(grid, (new_i, new_j), [0, netcode]) and \
                  cell_details[new_i][new_j].parent == (0, 0):

                if check_width_and_clearance(grid, (rows, columns), (i, j), dir_y, dir_x, [0, netcode], width, clearance) \
                    and not is_destination((new_i, new_j), (destination_row, destiantion_col)):
                    parent_cell = cell_details[i][j]
                    bend, bend_90_deg = 0, 0
                    parent_coord = parent_cell.parent

                    if check_bend(parent_coord, (i, j), (new_i, new_j)):
                        bend = 1
                        if check_90_deg_bend((new_i, new_j), (i, j), parent_coord):
                            bend_90_deg = 1
                              
                    g_new = parent_cell.g + h_euclidian((i, j), (new_i, new_j)) * 3  # greedy aprouch
                    h_new = h_euclidian((new_i, new_j), (destination_row, destiantion_col)) * 3
                    f_new = g_new + h_new + (bend_90_deg << 21) + (bend << 15)

                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        cell_details[new_i][new_j].update(f_new, h_new, g_new, (i, j))

                elif is_destination((new_i, new_j), (destination_row, destiantion_col)):
                    cell_details[new_i][new_j].parent = (i, j)
                    path = get_path()
                    return path
    return []



# Lee search == modified BFS
def lee_search(grid, grid_size: tuple, start: tuple[int, int], possible_ends: list,
               width: int, clearance: int, netcode: int):
    rows, columns = grid_size
    start_row, start_column = start

    if not is_valid((start_row, start_column), (rows, columns)):
        # print("\nStart invalid")
        return [], (-1, -1)
    
    values = [0, netcode] # 

    if not is_unblocked(grid, (start_row, start_column), values):
        # print("\nStart blocked")
        return [], (-1, -1)
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    mark_path_in_array = np.full(shape = (rows, columns), fill_value = -1, dtype = float)
    dest_x, dest_y = (-1, -1)

    def get_lee_path():
        path = []

        previous_direction = None  # Inițializează direcția anterioară ca fiind None
        best_y, best_x = dest_y, dest_x
        best_value = mark_path_in_array[best_y][best_x]

        current_x, current_y = best_x, best_y

        while True:
            path.append((best_y, best_x))
            if best_value == 0:     # start point
                path.reverse()
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
    mark_path_in_array[i][j] = 0
    visited.add((i, j))
    s = (i, j, 0.0) 
    q.append(s) 
    
    while len(q) > 0:
        entry = q.popleft()
        i, j, cost = entry
        mark_path_in_array[i][j] = cost
        if i != start_row or j != start_column:         
            if grid[i][j] == 0:
                for dest in possible_ends:
                    y, x = dest
                    if is_destination((i, j), (y, x)):
                        end_reached = True
                        break

            if end_reached:
                dest_y, dest_x = i, j
                #print(f"\nStart (pad) ({start_row}, {start_column}) reached")
                path = get_lee_path()
                return path

        # for each direction check the succesor
        for dir in directions:
            dir_y, dir_x = dir
            new_i = i + dir_y
            new_j = j + dir_x
            if is_valid((new_i, new_j), (rows, columns)) and is_unblocked(grid, (new_i, new_j), values) and \
                (new_i, new_j) not in visited:#mark_path_in_array[new_i][new_j] == -1: #(new_i, new_j) not in visited: #visited[new_i][new_j] == False:

                if check_width_and_clearance(grid, (rows, columns), (i, j), dir_y, dir_x, 
                                             [0, netcode, netcode + 0.5], width, clearance):
                    visited.add((new_i, new_j))
                    new_cost = cost + h_euclidian((i, j), (new_i, new_j))
                    q.append((new_i, new_j, new_cost))
    
    # print("\nDestination not reached")
    return [], (-1, -1)

# Apply Lee and A star to find tracks
def get_paths(template_grid, grid_shape: tuple[int, int, int], planned_routes: dict, routing_order: list, 
              pads: list = None):
    layers, rows, columns = grid_shape
    paths = []
    route_index = 0
    grid = copy.deepcopy(template_grid)

    aux_routes = copy.deepcopy(planned_routes)

    all_points = list(set(point for netcode in routing_order 
                          for point in aux_routes[netcode].coord_list +
                            [(x1, y1) for (x1, y1, x2, y2) in aux_routes[netcode].existing_conn] +
                            [(x2, y2) for (x1, y1, x2, y2) in aux_routes[netcode].existing_conn]
                        ))

    point_to_index = {point: idx for idx, point in enumerate(all_points)}
    uf = UnionFind(len(all_points))

    # Conex graph with points from existing_conn
    for netcode in routing_order:
        route = aux_routes[netcode]
        if route.existing_conn:
            for (x1, y1, x2, y2) in route.existing_conn:
                uf.union(point_to_index[(x1, y1)], point_to_index[(x2, y2)])
    
    for netcode in routing_order:
        route = aux_routes[netcode]
        nr_pads = len(route.original_coord)
        width = route.width
        clearance = route.clearance
        points  = route.coord_list
        other_points = list(set(point for point in route.existing_conn))
        original_coord = route.original_coord
        layer_id = route.layer_id # 0 - BOTTOM; 1 - TOP

        grid_copy = copy.deepcopy(grid[layer_id][:][:])
                
        route_index += 1
        
        if nr_pads == 2:
            for pad in pads:
                if pad.original_center in original_coord:
                    pad_clearance = pad.clearance_area
                    for point, w in pad_clearance.items():
                        y, x = point
                        for j in range(w):
                            grid_copy[y][x+j] = 0
                                          
            start_point = points[0]
            dest_point  = points[1]

            path = a_star_search(grid_copy, (rows, columns), start_point, dest_point, netcode, clearance, width)

        else:
            path = None
            dest_point = (-1, -1)
            def find_valid_start(points, other_points):
                all_points = list(set([(x1, y1) for (x1, y1, x2, y2) in other_points] + 
                                      [(x2, y2) for (x1, y1, x2, y2) in other_points] + points))
                # Sortăm punctele de start după mărimea componentei
                points_sorted = sorted(all_points, key=lambda point: uf.size[uf.find(point_to_index[point])])

                for point in points_sorted:
                    start_point_idx = point_to_index[point]
                    available_endpoints = [p for p in all_points if uf.find(point_to_index[p]) != uf.find(start_point_idx)]
                    if available_endpoints:
                        return point, start_point_idx, available_endpoints

                return None, None, None
            
            start_point, start_point_idx, available_endpoints = find_valid_start(points, other_points)

            if start_point and available_endpoints:
                for pad in pads:
                    if pad.center == start_point or pad.center in available_endpoints:
                        pad_clearance = pad.clearance_area
                        for point, w in pad_clearance.items():
                            y, x = point
                            for j in range(w):
                                grid_copy[y][x+j] = 0

                path, dest_point = lee_search(grid_copy, (rows, columns), start_point, available_endpoints, width, clearance,
                                                    netcode)   # momentan nu se va putea intersecta;  netcode + 0.7
        
        if dest_point != (-1, -1):
            aux_routes[netcode].existing_conn.append((start_point[0], start_point[1], dest_point[0], dest_point[1]))
            try:
                start_point_idx = point_to_index[start_point]
                dest_point_idx = point_to_index[dest_point]
                uf.union(start_point_idx, dest_point_idx)
            except Exception as e:
                ... # print(e)
            try:
                aux_routes[netcode].coord_list.remove(start_point)
                aux_routes[netcode].coord_list.remove(dest_point)
            except Exception as e:
                ... # print(e)
        
        elif dest_point == (-1, -1) or not path:
            # print("Unplaced route")
            paths.append(Path(start_point, dest_point, netcode, [], width, clearance, False, None, layer_id))
            continue
        
        grid[layer_id][:][:] = mark_path_in_array(grid[layer_id][:][:], path, netcode)
        grid[layer_id][:][:] = mark_adjent_path(grid[layer_id][:][:], (rows, columns), path, width, netcode)
        grid[layer_id][:][:] = mark_clearance_on_grid(grid[layer_id][:][:], (rows, columns), path, width, clearance, netcode)

        path_found = Path(start_point, dest_point, netcode, path, width, clearance, False, None, layer_id)
        paths.append(path_found)

        for pad in pads:
            if pad.center == dest_point or pad.center == start_point:
                pad_clearance = pad.clearance_area
                for point, w in pad_clearance.items():
                    y, x = point
                    for j in range(w):
                        grid[layer_id][y][x+j] = -1
    
    return paths
import copy
from random import shuffle, randint, choice
import random
from utils import Individual, Path, update_grid_with_paths, mark_path_in_array, check_90_deg_bend
from routing_algorithms import get_paths, a_star_search, get_perpendicular_direction
import os
import psutil
import numpy as np

def generate_individual(args):
    try:
        seed, ordered_indexes, starting_grid, template_grid, planned_routes, pads_list, netcodes_list, layers, rows, columns = args
        random.seed(seed)
        if not ordered_indexes:
            random_order_indexes = copy.deepcopy(netcodes_list)
            shuffle(random_order_indexes)
        else:
            random_order_indexes = ordered_indexes

        if np.all(np.array(starting_grid) != None):
            possible_solution = get_paths(starting_grid, (layers, rows, columns), planned_routes, random_order_indexes, pads_list)
        else:
            possible_solution = get_paths(template_grid, (layers, rows, columns), planned_routes, random_order_indexes, pads_list)

        individual = Individual()
        individual.set_values(order=random_order_indexes, paths=possible_solution)
        return individual
    except Exception as e:
        print(f"Exception in generate_individual: {e}")
        import traceback
        traceback.print_exc()
        return None


# operations used for crossover; it is applied directly onto the children
def crossover_order(seed, child1_order, child2_order, crossover_points: int = 1):
    random.seed(seed)
    size = len(child1_order)
    index = randint(0, (size - 1) // 2)

    for _ in range(crossover_points):
        aux1_order = copy.deepcopy(child1_order)
        aux2_order = copy.deepcopy(child2_order)
        child1_order = aux1_order[:index] + aux2_order[index:]
        child2_order = aux2_order[:index] + aux1_order[index:]
        index = randint(index, size - 1)

    return child1_order


def get_similarity_index(list1, list2) -> int:
    for i in range(min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            return i-1
    return min(len(list1), len(list2))


# Modify process / subprocess priority
def set_high_priority():
    p = psutil.Process(os.getpid())
    try:
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    except AttributeError:
        # On UNIX, use 'os' directly
        os.nice(-20)


# Worker initialization function
def init_worker():
    set_high_priority()



def crossover(seed, crossover_points, args):
    parent1, parent2, template_grid, planned_routes, pads_list, LAYERS, ROWS, COLUMNS = args
    random.seed(seed)

    parent_order1 = parent1.order
    parent_order2 = parent2.order

    seed1 = random.random()

    aux_seed = random.random()
    child_order1 = crossover_order(aux_seed, parent_order1, parent_order2, crossover_points)
    if len(parent1.paths) != len(parent2.paths):
        random.seed(aux_seed)
        return generate_individual((seed1, child_order1, None, template_grid, planned_routes, child_order1, LAYERS, ROWS, COLUMNS))

    index1 = get_similarity_index(parent_order1, child_order1)

    grid1 = copy.deepcopy(template_grid)
    previous_paths_parent1 = None
    if index1 >= 0:
        previous_paths_parent1 = parent1.paths[:index1]
        grid1 = update_grid_with_paths(grid1, (ROWS, COLUMNS), previous_paths_parent1)
    else:
        return parent1, parent2

    aux_planned_routes1 = copy.deepcopy(planned_routes)


    if previous_paths_parent1:
        for path_object in previous_paths_parent1:
            if path_object and path_object.path:
                aux_planned_routes1[path_object.netcode].existing_conn.append(
                    (path_object.start[0], path_object.start[1], path_object.destination[0], path_object.destination[1])
                )

    slice_order1 = parent1.order[index1:]
    new_paths1 = get_paths(grid1, (LAYERS, ROWS, COLUMNS), aux_planned_routes1, slice_order1, pads_list)
    
    # Step 4: Combine initial and new paths to create children
    child1 = Individual()
    child1_path = previous_paths_parent1 + new_paths1 if previous_paths_parent1 else new_paths1
    child1.set_values(parent1.order, child1_path)

    return child1


# Alter path from the list of routes found for an individual
def path_mutation(parent: Individual, args):    
    order = copy.deepcopy(parent.order)
    NR_PATH_PLANNED, template_grid, planned_routes, pads_list, LAYERS, ROWS, COLUMNS = args
    path_index = randint(0, NR_PATH_PLANNED-1)
    
    if path_index < len(parent.paths):
        mutated_path = parent.paths[path_index]
        current_path = mutated_path.path
        length = len(current_path)
        if length < 10:
            return parent
        
        # Mark the old paths (unmodified) in template grid 
        grid = copy.deepcopy(template_grid)
        previous_paths = parent.paths[0 : path_index]
        if previous_paths:
            grid = update_grid_with_paths(grid, (ROWS, COLUMNS), previous_paths)

        layer = mutated_path.layer_id
        width = mutated_path.width
        clearance = mutated_path.clearance
        netcode = mutated_path.netcode

        deviation = randint(1, 5)
        deviation_sign = choice([-1, 1])

        start_i = randint(3, length-6)
        end_i = randint(start_i + 2, length - 3)
        
        (start_y_aux1, start_x_aux1), (start_y_aux2, start_x_aux2) = current_path[start_i-2], current_path[start_i-1]
        start_y, start_x = current_path[start_i]
        
        (end_y_aux1, end_x_aux1), (end_y_aux2, end_x_aux2) = current_path[end_i+2], current_path[end_i+1]
        end_y, end_x = current_path[end_i]


        # start
        if not check_90_deg_bend((start_y, start_x), (start_y_aux2, start_x_aux2), (start_y_aux1, start_x_aux1)):
            dir_y_start, dir_x_start = start_y_aux2 - start_y_aux1, start_x_aux2 - start_x_aux1
        else:   
            dir_y_start, dir_x_start = start_y - start_y_aux1, start_x - start_x_aux1
            dir_perp_y_start, dir_perp_x_start = get_perpendicular_direction(dir_y_start, dir_x_start)
            dir_y_start, dir_x_start = dir_perp_y_start, dir_perp_x_start

        # end
        if not check_90_deg_bend((end_y, end_x), (end_y_aux2, end_x_aux2), (end_y_aux1, end_x_aux1)):
            dir_y_end, dir_x_end = end_y_aux2 - end_y_aux1, end_x_aux2 - end_x_aux1
        else:   
            dir_y_end, dir_x_end = end_y - end_y_aux2, end_x - end_x_aux2
            dir_perp_y_end, dir_perp_x_end = get_perpendicular_direction(dir_y_end, dir_x_end)
            dir_y_end, dir_x_end = dir_perp_y_end, dir_perp_x_end


        grid[layer][start_y][start_x] = -1
        grid[layer][end_y][end_x] = -1

        start_list = []
        end_list   = []

        for j in range(deviation):
            p1_y, p1_x = start_y_aux1 + j * deviation_sign * dir_y_start, start_x_aux1 + j * deviation_sign * dir_x_start
            p2_y, p2_x = end_y_aux2   + j * deviation_sign * dir_y_start, end_x_aux2   + j * deviation_sign * dir_x_start

            if grid[layer][p1_y][p1_x] == 0 or grid[layer][p2_y][p2_x] == 0:
               grid[layer][p1_y][p1_x] = -1
               grid[layer][p2_y][p2_x] = -1
            else:
                p1_y, p1_x = p1_y - deviation_sign * dir_y_start, p1_x - deviation_sign * dir_x_start
                p2_y, p1_x = p2_y - deviation_sign * dir_y_start, p2_x - deviation_sign * dir_x_start
                break

            start_list.append((p1_y, p1_x))
            end_list.append((p2_y, p2_x))
        
        grid[layer][p1_y][p1_x] = 0
        grid[layer][p2_y][p2_x] = 0

        # Using a* find the shortest path to connect the new points
        deviation_path = a_star_search(grid[layer][:][:], (ROWS, COLUMNS), (p1_y, p1_x), (p2_y, p2_x), netcode, clearance, width)
        if not deviation_path:
            return parent

        current_path = current_path[:start_i] + start_list + deviation_path + end_list + current_path[end_i+1:]
        grid[layer][:][:] = mark_path_in_array(grid[layer][:][:], current_path, netcode)
        updated_path = Path(mutated_path.start, mutated_path.destination, netcode, current_path, width, clearance, True, None, layer)
        
        paths = previous_paths + [updated_path] if previous_paths else [updated_path]
        aux_planned_routes = copy.deepcopy(planned_routes)
        for path_object in (paths):
            if path_object and path_object.path:
                netcode = path_object.netcode
                aux_planned_routes[netcode].existing_conn.append((path_object.start[0], path_object.start[1],
                                                                   path_object.destination[0], path_object.destination[1]))

        # Find the rest of paths
        slice_order = order[path_index+1:]
        new_paths = get_paths(grid, (LAYERS, ROWS, COLUMNS), aux_planned_routes, slice_order, pads_list)

        # Form a child from old, modified and new paths
        child = Individual()
        child.set_values(order, paths + new_paths)
        return child
    
    return parent


# Modify the routing order for an individual
def order_mutation(seed, parent: Individual, args):
    order = copy.deepcopy(parent.order)
    NR_PATH_PLANNED, template_grid, planned_routes, pads_list, netcodes_list, LAYERS, ROWS, COLUMNS = args

    random.seed(seed)
    index1 = randint(0, NR_PATH_PLANNED-1)
    index2 = randint(0, NR_PATH_PLANNED-1)

    while index1 == index2:
        index2 = randint(0, NR_PATH_PLANNED-1)
    
    index = min(index1, index2)
    order[index1], order[index2] = order[index2], order[index1] # swap the order for routing

    if len(parent.paths) > index:
        previous_paths_parent = parent.paths[:index]
        grid = update_grid_with_paths(template_grid, (ROWS, COLUMNS), previous_paths_parent)

        aux_planned_routes = copy.deepcopy(planned_routes)

        for path_object in previous_paths_parent:
            if path_object:
                aux_planned_routes[path_object.netcode].existing_conn.append(
                    (path_object.start[0], path_object.start[1], path_object.destination[0], path_object.destination[1])
                )
                    
        slice_order = parent.order[index:]
        
        new_paths = get_paths(grid, (LAYERS, ROWS, COLUMNS), aux_planned_routes, slice_order, pads_list)
                
        # Step 4: Combine initial and new paths to create children
        child = Individual()
        child.set_values(parent.order, previous_paths_parent + new_paths)
        return child
    else:
        return generate_individual(seed, None, None, template_grid, planned_routes, pads_list, netcodes_list, LAYERS, ROWS, COLUMNS)
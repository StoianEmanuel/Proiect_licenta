import sys
import numpy as np
from math import sqrt, ceil
import re
sys.path.append("C:\\Program Files\\KiCad\\7.0\\bin")

import pcbnew # type: ignore
from pcbnew import ToMM # type: ignore
from utils import Pad, UserSettings, PlannedRoute, UnionFind, division_int, mark_path_in_array
from routing_algorithms import mark_clearance_on_grid, get_paths, mark_adjent_path
from joblib import Parallel, delayed

# Board --> GetNetInfoList --> parcurgere  / GetNetInfoItem(netcode) --> NetListItem --> GetNetname

def remove_useless_routes(routes: dict, pads: list):
    # Inițializăm structura Union-Find pentru a ține evidența componentelor conexe
    unique_points = set()
    if routes:
        for route in routes.values():
            for start_y, start_x, end_y, end_x in route.coord_list:
                unique_points.add((start_y, start_x))
                unique_points.add((end_y, end_x))
        point_to_index = {point: idx for idx, point in enumerate(unique_points)}
        uf = UnionFind(len(unique_points))

        # Creăm componentele conexe pentru punctele din trasee
        for route in routes.values():
            for start_y, start_x, end_y, end_x in route.coord_list:
                start_point_idx = point_to_index[(start_y, start_x)]
                dest_point_idx = point_to_index[(end_y, end_x)]
                uf.union(start_point_idx, dest_point_idx)

        # Inițializăm o listă cu punctele asociate pad-urilor
        pad_list = [pad.center for pad in pads]

        # Identificăm traseele inutile și le stocăm într-un set
        useless = set()
        for point in unique_points:
            ok = False
            id_p = point_to_index[point]
            for center in pad_list:
                try:
                    id_c = point_to_index[center]
                    if uf.find(id_p) == uf.find(id_c): 
                        ok = True
                        break
                except KeyError:
                    pass
            if not ok:
                useless.add(point)

        # Eliminăm traseele inutile din dicționarul de trasee
        filtered_routes = {}
        for netcode, route in routes.items():
            coord_list = route.coord_list
            aux_c_list = []
            aux_o_list = []
            for index in range(len(coord_list)):
                start_y, start_x, end_y, end_x = coord_list[index]
                if (start_y, start_x) in useless or (end_y, end_x) in useless:
                    continue
                else:
                    aux_c_list.append((coord_list[index]))
                    aux_o_list.append((route.original_coord[index]))
            if aux_c_list:
                filtered_routes[netcode] = PlannedRoute(netcode, route.netname, route.width, route.clearance,
                                                        aux_c_list, aux_o_list, aux_o_list)
                
        return filtered_routes


# if keep_values -- keep
# if overwrite == True - check for pwr, gnd and then all
# else assign default value for
def get_settings_from_existing_tracks(board: pcbnew.BOARD, user_settings: UserSettings = None):
    tracks = board.GetTracks()       
    max_unit = 100000  #
    routes = {}
    
    connections = {}
    for track in tracks:
        netcode = track.GetNetCode()
        netname = track.GetNetname()
        layer   = track.GetLayerName()
        start_x, start_y = track.GetStart() # original nm; GetStart -> x, y
        end_x, end_y = track.GetEndX(), track.GetEndY() # nm
        width = track.GetWidth()   # nm
        clearance = track.GetOwnClearance(pcbnew.B_Cu, track.GetStart()) # Back copper layer ; nm

        # Suprascrie valorile cu cele personalizate
        for settings in user_settings.dict.values():
            pattern = settings['pattern']
            if pattern.search(netname):
                width = settings['width']
                clearance = settings['clearance']
                layer = settings['layer_id']
                break

        if netcode not in routes.keys():
            routes[netcode] = PlannedRoute(
                netcode = netcode,
                netname = netname,
                width = width,
                clearance = clearance,
                coord_list = [(start_y, start_x, end_y, end_x)],
                original_coord = [(start_y, start_x, end_y, end_x)],
                layer_id = 1 if layer == 'F.Cu' and user_settings.layers == 2 else 0
            )
            if clearance % 10000 !=0 or width % 10000 != 0:
                max_unit = 1000
            elif clearance % 100000 !=0 or width % 100000 != 0:
                max_unit = 10000
            connections[netcode] = {'conns': [(start_y, start_x, end_y, end_x)]}
        else:
            routes[netcode].add_track((start_y, start_x, end_y, end_x))
            routes[netcode].original_coord.append((start_y, start_x, end_y, end_x))
            connections[netcode]['conns'].append((start_y, start_x, end_y, end_x))
    
    max_unit = min(user_settings.get_multiplier(), max_unit)
    
    for netcode, route in routes.items():
        route.width = division_int(route.width, max_unit)
        route.clearance = division_int(route.clearance, max_unit)
        route.coord_list = [tuple(division_int(coord, max_unit) for coord in track) for track in route.coord_list]
    
    if routes:
        return routes, max_unit
    
    return None, max_unit


# Get center coord in MM for pcb
def get_board_center(board: pcbnew.BOARD):  # checked
    try:
        center_x, center_y = board.GetFocusPosition()
        return center_y, center_x
    except Exception as e:
        print(e)
    return 105000000, 148501000 # in nm; (297.002/2, 210.000/2) mm A4 sheet size


# Get size of pcb in MM from its bounding box
def get_board_size(board: pcbnew.BOARD):    # checked
    try:
        bounding_box = board.GetBoundingBox()
        width = bounding_box.GetWidth()
        height = bounding_box.GetHeight()
        return height, width
    except Exception as e:
        print(e)
    return 210000000, 297002000 # in nm; (297.002, 210.000) mm A4 sheet size


# Input a list of coordinates (for trace, pad) and a value represeting the offset that will be added / substracted
def remove_offset_coord(coord_list, offset_y, offset_x):
    updated_list = [(y - offset_y, x - offset_x) for (y, x) in coord_list]
    return updated_list


def get_clearance_shape(original_shape, clearance, remove_inner = True):
    clearance_shape = set()

    # Adăugăm punctele de clearance extinse
    for y, x in original_shape:
        for dy in range(-clearance, clearance + 1):
            for dx in range(-clearance, clearance + 1):
                clearance_shape.add((y + dy, x + dx))

    # Eliminăm punctele originale din clearance_shape
    if remove_inner:
        for point in original_shape:
            clearance_shape.discard(point)

    # Convertim în listă și sortăm
    clearance_shape = list(clearance_shape)
    clearance_shape.sort()

    return clearance_shape



def get_shape_from_vertices(vertices_list):
    # Sortează vertecșii după Y apoi X
    vertices_list.sort(key=lambda x: (x[1], x[0])) 

    # Extrage valorile minime și maxime pentru y
    y_values = [vertex[1] for vertex in vertices_list]
    min_y, max_y = min(y_values), max(y_values)
    
    shape_points = []
    prev_x_points = []

    # Parcurge fiecare rând (y) de la min_y la max_y
    for y in range(min_y, max_y + 1):
        # Punctele active pentru rândul curent
        current_x_points = [vertex[0] for vertex in vertices_list if vertex[1] == y]
        current_x_points.sort()

        # Dacă nu există puncte active pe acest rând, copiază punctele active de pe rândul anterior
        if not current_x_points:
            current_x_points = prev_x_points

        # Parcurge x de la min_x la max_x și adaugă punctele margine
        is_edge = False
        for x in range(min(current_x_points), max(current_x_points) + 1):
            if x in current_x_points:
                is_edge = not is_edge
            if is_edge or x in current_x_points:
                shape_points.append((y, x))

        prev_x_points = current_x_points

    return shape_points


def process_polygon(polygon, multiplier = 100):
    vertices_list = [(division_int(polygon.CVertex(i)[0], multiplier), division_int(polygon.CVertex(i)[1], multiplier)) 
                        for i in range(polygon.VertexCount())]
    return vertices_list


def get_pads_parallel(board: pcbnew.BOARD, multiplier = 100, max_jobs = 3):
    pads = []
    
    def process_pad(pad):
        center_x, center_y = pad.GetPosition()
        original_center = (center_y, center_x)  # nm

        center_x, center_y = division_int(center_x, multiplier), division_int(center_y, multiplier)
        netcode = pad.GetNetCode()
        netname = pad.GetNetname()
        polygon = pad.GetEffectivePolygon()
        clearance = division_int(pad.GetOwnClearance(pcbnew.F_Cu), multiplier)
        vertices_list = process_polygon(polygon, multiplier)
        pad_area = get_shape_from_vertices(vertices_list=vertices_list)
        clearance_area = get_clearance_shape(pad_area, clearance)

        return Pad(center=(center_y, center_x), original_center=original_center,
                    pad_area=pad_area, netcode=netcode, netname=netname,
                    clearance=clearance, clearance_area=clearance_area)
    
    footprints = board.GetFootprints()
    count_pad = sum(len(part.Pads()) for part in footprints)
    jobs = min(max_jobs, count_pad/4)
    pads = Parallel(n_jobs=jobs)(delayed(process_pad)(pad) for part in footprints for pad in part.Pads())

    return pads


# Returns list with verteces assigend to rule_areas (areas to avoid)
def get_rule_areas(board: pcbnew.BOARD, multiplier = 1000):        # checked
    areas = []
    zones = board.Zones()
    for zone in zones:
        if zone.GetLayer() == pcbnew.B_Cu:  # routes are currently placed only on B_Cu layer; posibilty for F_Cu
            polygon = zone.Outline()
            vertices_list = process_polygon(polygon, multiplier)
            rule_area = get_shape_from_vertices(vertices_list)
            areas.append(rule_area)
    return areas


def get_interval(area, current_max_x, current_max_y, current_min_x, current_min_y): # area = list of tuple coord
    aux = np.array(area)
    min_x, min_y = np.min(aux[:, 1]), np.min(aux[:, 0])
    max_x, max_y = np.max(aux[:, 1]), np.max(aux[:, 0])
    current_min_x, current_min_y = min(min_x, current_min_x), min(min_y, current_min_y)
    current_max_x, current_max_y = max(max_x, current_max_x), max(max_y, current_max_y)
    return current_max_x, current_max_y, current_min_x, current_min_y


# TODO
def get_connected_components(existing_tracks):
    # Inițializăm structura Union-Find pentru a ține evidența componentelor conexe
    unique_points = set()
    for route in existing_tracks.values():
        for start_y, start_x, end_y, end_x in route.coord_list:
            unique_points.add((start_y, start_x))
            unique_points.add((end_y, end_x))
    point_to_index = {point: idx for idx, point in enumerate(unique_points)}
    uf = UnionFind(len(unique_points))

    # Creăm componentele conexe pentru punctele din trasee
    for route in existing_tracks.values():
        for start_y, start_x, end_y, end_x in route.coord_list:
            start_point_idx = point_to_index[(start_y, start_x)]
            dest_point_idx = point_to_index[(end_y, end_x)]
            uf.union(start_point_idx, dest_point_idx)

    # Inițializăm o listă cu toate punctele din trasee
    all_points = list(unique_points)

    # Inițializăm o listă de liste pentru a stoca conexiunile pentru fiecare netcode
    connected_components = {netcode: [] for netcode in existing_tracks}

    # Identificăm conexiunile pentru fiecare netcode
    for netcode, route in existing_tracks.items():
        for start_y, start_x, end_y, end_x in route.coord_list:
            start_point_idx = point_to_index[(start_y, start_x)]
            dest_point_idx = point_to_index[(end_y, end_x)]
            parent_start = uf.find(start_point_idx)
            parent_dest = uf.find(dest_point_idx)
            if parent_start == parent_dest:
                # Ignorăm conexiunile între același tată și destinație
                continue
            # Adăugăm conexiunea la lista de conexiuni pentru netcode-ul respectiv
            connected_components[netcode].append((parent_start, parent_dest))

    return connected_components





def get_netcodes(board, user_settings, existing_tracks = None):
    netcode_dict = {}
    clock_pattern = re.compile(r'(?i)\b(clock|clk)\b')

    for pad in board.GetPads():
        netcode = pad.GetNetCode()
        center_x, center_y = pad.GetCenter()
        center = (center_y, center_x)

        if netcode not in netcode_dict:
            netname = pad.GetNetname()
            netcode_dict[netcode] = {
                'frequency': 1,
                'netname': netname,
                'coord_list': [center],
                'clearance': None,
                'width': None,
                'layer_id': None
            }

            if clock_pattern.search(netname):
                netcode_dict[netcode]['clearance'] = user_settings['ALL']['clearance']
                netcode_dict[netcode]['width'] = int(user_settings['ALL']['clearance'] * 1.5)
                netcode_dict[netcode]['layer_id'] = 1 if user_settings.layers == 2 else 0
            # trebuie adaugata logica pentru a evita track-urile deja amplasate
            elif existing_tracks and netcode in existing_tracks and netname in existing_tracks[netcode].netname:
                netcode_dict[netcode]['clearance'] = existing_tracks[netcode].clearance
                netcode_dict[netcode]['width'] = existing_tracks[netcode].width
                netcode_dict[netcode]['layer_id'] = existing_tracks[netcode].layer_id
            else:
                for value in user_settings.dict.values():
                    pattern = value['pattern']
                    if pattern.search(netname):
                        netcode_dict[netcode]['clearance'] = value['clearance']
                        netcode_dict[netcode]['width'] = value['width']
                        netcode_dict[netcode]['layer_id'] = 1 if value['layer_id'] == 'F.Cu' and user_settings.layers == 2 else 0
                        break

        else:
            netcode_dict[netcode]['frequency'] += 1
            netcode_dict[netcode]['coord_list'].append(center)

    return netcode_dict       


def update_coord_for_net_dict(netcode_dict, offset_x, offset_y):
    for values in netcode_dict.values():
        coord_list = values['coord_list']
        aux = remove_offset_coord(coord_list, offset_x, offset_y)
        values['coord_list'] = aux


def get_total_width(netcode_dict: dict) -> int: 
    width = 0
    for value in netcode_dict.values():
        frequency = value['frequency'] - 1
        width += frequency*(value['clearance'] + value['width'])
    return width


def get_template_grid(grid_shape, rule_areas, pads):
    try:
        grid = np.zeros(grid_shape, dtype=float)
        z = grid_shape[0]
        for i in range(z):
            if rule_areas:
                for area in rule_areas:
                    if area:
                        for coord in area:
                            grid[i][coord[0], coord[1]] = -1

            for pad in pads:
                for coord in pad.pad_area:
                    grid[i][coord[0], coord[1]] = -1
                for coord in pad.clearance_area:
                    grid[i][coord[0], coord[1]] = -1 
    except Exception as e:
        print(e)
    return grid



# Offset for x and y -- helps reducing grid size
def get_board_offset(board_center, board_size, pads, rule_areas, netcode_dict):
    center_y, center_x = board_center
    height, width = board_size
    origin_x, origin_y = center_x - width // 2, center_y - height // 2
    end_x, end_y = center_x + width // 2, center_y + height // 2

    max_part_x, max_part_y = float('-inf'), float('-inf')
    min_part_x, min_part_y = float('inf'), float('inf')

    if rule_areas:
        for area in rule_areas:
            if area:
                max_part_x, max_part_y, min_part_x, min_part_y = get_interval(area, max_part_x, max_part_y, min_part_x, min_part_y)

    for pad in pads:
        area = pad.clearance_area
        max_part_x, max_part_y, min_part_x, min_part_y = get_interval(area, max_part_x, max_part_y, min_part_x, min_part_y)

    # max_clearance = board.GetDesignSettings().GetBiggestClearanceValue()
    dist_kept_for_tracks = int(get_total_width(netcode_dict = netcode_dict)) # on board edges

    offset_lt_x = max(origin_x, min_part_x - dist_kept_for_tracks)
    offset_lt_y = max(origin_y, min_part_y - dist_kept_for_tracks)

    border_rb_x = min(max_part_x + dist_kept_for_tracks, end_x) # right bottom
    border_rb_y = min(max_part_y + dist_kept_for_tracks, end_y)

    max_x = border_rb_x - offset_lt_x + 1
    max_y = border_rb_y - offset_lt_y + 1
    return offset_lt_y, offset_lt_x, max_y, max_x


# remove netcodes (+tracks) from netcode_dict if frequency = 1 (connected to one pad)
def remove_unusable_netcodes(netcode_dict):
    new_dict = {key: values for key, 
            values in netcode_dict.items() if values['frequency'] != 1}
    return new_dict


def update_values(pads, offset_y = 0, offset_x = 0, rule_areas = None):
    for index in range(len(pads)):
        pads[index].center = (pads[index].center[0] - offset_y, pads[index].center[1] - offset_x)

        aux = remove_offset_coord(pads[index].pad_area, offset_y, offset_x)
        pads[index].pad_area = aux

        aux = remove_offset_coord(pads[index].clearance_area, offset_y, offset_x)
        pads[index].clearance_area = aux


    if rule_areas:
        for index in range(len(rule_areas)):
            if rule_areas[index]:
                aux = remove_offset_coord(rule_areas[index], offset_y, offset_x)
                rule_areas[index] = aux



def see_changes():
    pcbnew.Refresh()


def get_path_from_2verteces(start, end):
    main_path = []
    start_y, start_x = start
    end_y, end_x = end

    # Cazul în care punctele sunt pe aceeași linie orizontală
    if start_y == end_y:
        # Adăugăm punctele de-a lungul liniei orizontale
        for x in range(min(start_x, end_x), max(start_x, end_x) + 1):
            main_path.append((start_y, x))
        return main_path
    
    # Cazul în care punctele sunt pe aceeași linie verticală
    if start_x == end_x:
        # Adăugăm punctele de-a lungul liniei verticale
        for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
            main_path.append((y, start_x))
        return main_path

    # Calculăm panta și interceptarea
    m = (end_y - start_y) / (end_x - start_x)
    c = start_y - m * start_x
    
    if start_y > end_y:
        start_y, end_y = end_y, start_y
        start_x, end_x = end_x, start_x
    
    # Inițializăm punctul anterior
    x_prev, y_prev = start_x, start_y
    
    for y in range(start_y, end_y + 1):
        x = (y - c) / m  # Calculăm x folosind ecuația dreptei
        
        # Verificăm dacă trebuie să adăugăm un punct intermediar
        distanta = sqrt((x - x_prev)**2 + (y - y_prev)**2)
        if distanta > sqrt(2):
            puncte_intermediare = ceil(distanta / sqrt(2)) - 1
            dx = (x - x_prev) / (puncte_intermediare + 1)
            dy = (y - y_prev) / (puncte_intermediare + 1)
            for i in range(1, puncte_intermediare + 1):
                main_path.append((int(y_prev + i*dy), int(x_prev + i*dx)))
        
        if not main_path or main_path[-1] != (y, int(x)):
            main_path.append((y, int(x)))  # Rotunjim x la cea mai apropiată valoare întreagă
        
        x_prev, y_prev = x, y
        
    return main_path
    

# TODO  ------------------  ---
def get_nets_from_netcode_dict(netcode_dict, existing_conns = None):
    if not existing_conns:
        existing_conns = {}
    
    planned_routes = {}
    routing_order = []
    
    connected_nodes = get_connected_components(existing_conns)

    for key, values in netcode_dict.items():
        width = values['width']
        clearance = values['clearance']
        netcode = key
        netname = values['netname']
        nodes = values['coord_list']
        nr_tracks = values['frequency']
        layer_id = values['layer_id']
    
        connections = []
        scaled_connections = []
        if existing_conns and netcode in existing_conns.keys():
            for index in range(len(nodes)):
                if nodes[index] in connected_nodes[netcode]:
                    connections.append(nodes[index])
                    scaled_connections.append()
                
            existing_conn = existing_conns[netcode].coord_list
        else:
            existing_conn = []
        
        planned_routes[netcode] = PlannedRoute(
                    netcode = netcode,
                    netname = netname,
                    width = width,
                    clearance = clearance,
                    coord_list = nodes,
                    original_coord = nodes,
                    existing_conn = existing_conn,
                    layer_id = layer_id
                )
        while nr_tracks > 1:
            routing_order.append(netcode)
            nr_tracks -= 1
    
    return planned_routes, routing_order


def remove_offset_in_list(points, offset_y, offset_x, multiplier = None):
    aux = []
    for point in points:
        if len(point) == 2:
            y, x = point
            y = division_int(y, multiplier) - offset_y
            x = division_int(x, multiplier) - offset_x
            aux.append((y, x))
        else:
            y1, x1, y2, x2 = point
            y1, x1 = y1 - offset_y, x1 - offset_x
            y2, x2 = y2 - offset_y, x2 - offset_x
            aux.append((y1, x1, y2, x2))
    return aux


def update_coord_in_nets(planned_routes, routing_order, offset_y, offset_x, multiplier):
    unique_netcodes = np.unique(routing_order)
    for netcode in unique_netcodes:
        coord_list = planned_routes[netcode].coord_list
        aux = remove_offset_in_list(coord_list, offset_y, offset_x, multiplier)
        planned_routes[netcode].coord_list = aux

        existing_conn = planned_routes[netcode].existing_conn
        aux = remove_offset_in_list(existing_conn, offset_y, offset_x)
        planned_routes[netcode].existing_conn = aux
        


def revert_coord(coord_list, offset_y, offset_x, multiplier):
    return [(p[0] * multiplier + offset_y, p[1] * multiplier + offset_x) for p in coord_list]



def swap_order(coordinate: tuple[int, int]):
    return coordinate[1], coordinate[0]


def mark_existing_tracks_on_grid(grid, grid_shape, existing_tracks, offset_y = 0, offset_x = 0):
    z, max_y, max_x = grid_shape
    for key in existing_tracks.keys():
        track = existing_tracks[key]
        point_list = track.coord_list
        netcode = track.netcode # key should be == netcode
        width = track.width
        clearance = track.clearance
        layer_id = track.layer_id
        for point_set in point_list:
            start_y, start_x, end_y, end_x = point_set
            l = remove_offset_coord([(start_y, start_x), (end_y, end_x)], offset_y, offset_x)
            start_y, start_x = l[0]
            end_y, end_x = l[1]

            path = get_path_from_2verteces((start_y, start_x), (end_y, end_x))
            grid[layer_id][:][:] = mark_path_in_array(grid[layer_id][:][:], path, key)

            grid[layer_id][:][:] = mark_adjent_path(grid[layer_id][:][:], (max_y, max_x), path, width, key)

            grid[layer_id][:][:] = mark_clearance_on_grid(grid[layer_id][:][:], (max_y, max_x), path, width, clearance, key)

    return grid



def get_original_coord_for_path(path, nets):
    start_tf = path.start
    dest_tf  = path.destination
    for netcode, values in nets.items():
        if path.netcode == netcode:
            original_coords = values.original_coord
            transformed_coords = values.coord_list
            
            for index in range(len(original_coords)):
                if start_tf == transformed_coords[index]:
                    start_og = original_coords[index]
                if dest_tf  == transformed_coords[index]:
                    dest_og  = original_coords[index]
                
            return start_og, dest_og


def format_float(value):
    return "{:.3f}".format(value)


def truncate_value(f, decimals):
    s = '{}'.format(f)
    i, p, d = s.partition('.')
    return float('.'.join([i, (d+'0'*decimals)[:decimals]]))



def get_cleaned_path(path):
    cleaned_path = []
    cleaned_path.append(path[0])
    if len(path) >= 3:
        for index in range(1, len(path)-1):
            previous_value = path[index-1]
            current_value  = path[index]
            next_value = path[index+1]
            prev_incr_x, prev_incr_y = previous_value[0] - current_value[0], previous_value[1] - current_value[1]
            next_incr_x, next_incr_y = current_value[0]  - next_value[0]   , current_value[1]  - next_value[1]
            if (prev_incr_x, prev_incr_y) != (next_incr_x, next_incr_y):
                cleaned_path.append(current_value)
    cleaned_path.append(path[-1])      
    return cleaned_path



def get_segments_for_board(paths_found, nets, offset_board: tuple[int, int], multiplier):
    segments = set()
    # net_items = board.GetNetsByNetcode()
    offset_y, offset_x = offset_board
    for path in paths_found:
        if path:
            layer_str = f'(layer "B.Cu")' if path.layer_id == 0 else f'(layer "F.Cu")'
            original_start, original_dest = get_original_coord_for_path(path, nets)

            simplified_path     = path.simplified_path # get NETITEM by netcode
            y, x = (simplified_path[0][0] + offset_y) * multiplier, (simplified_path[0][1] + offset_x) * multiplier
            aprox_error_y, aprox_error_x  = y - original_start[0], x - original_start[1]

            original_start = truncate_value(ToMM(int(original_start[0])), 3), truncate_value(ToMM(int(original_start[1])), 3)
            simplified_path[0]  = swap_order(original_start)

            original_dest  = truncate_value(ToMM(int(original_dest[0])), 3),  truncate_value(ToMM(int(original_dest[1])), 3)
            simplified_path[-1] = swap_order(original_dest)
            
            for index in range(1, len(simplified_path)-1):
                y, x = simplified_path[index]
                y = truncate_value(ToMM(int((y + offset_y) * multiplier + aprox_error_y)), 3)
                x = truncate_value(ToMM(int((x + offset_x) * multiplier + aprox_error_x)), 3)
                simplified_path[index] = (x, y)


            cleaned_path = get_cleaned_path(simplified_path)

            netcode = path.netcode
            net_str = f"(net {netcode})"

            width     = ToMM(path.width * multiplier)
            width_str = f"(width {format_float(width)})"

            for index in range(len(cleaned_path)-1):
                start = cleaned_path[index]
                start_str = f"(start {format_float(start[0])} {format_float(start[1])})"

                end = cleaned_path[index+1]
                end_str = f"(end {format_float(end[0])} {format_float(end[1])})"

                segments.add(f"  (segment {start_str} {end_str} {width_str} {layer_str} {net_str})")

    return segments



def write_segments_to_EOF(filepath, segments_list, delete_previous_tracks = True):
    # Citeste continutul fisierului
    with open(filepath, 'r') as f:
        content = f.readlines()

    closing_paranthesis_count = 0
    index1, index2 = None, None
    for index in range(len(content)-1, -1, -1):
        line = content[index].strip()
        if line == ')':
            closing_paranthesis_count += 1
            index1 = index
            if delete_previous_tracks:
                if closing_paranthesis_count == 1:
                    index2 = index
                if closing_paranthesis_count == 2:
                    break

    # Construieste string-urile pentru segmente
    segments_string = '\n'.join(segments_list) + '\n'

    # Adauga segmentele la fisier
    if delete_previous_tracks:
        content = content[:index1+1] + [segments_string] + content[index2:]
    else:
        content = content[:index2] + [segments_string] + content[index2:]

    with open(filepath, 'w') as f:
        f.writelines(content)



def get_board_dimensions(board: pcbnew.BOARD, multiplier: int):
    size = get_board_size(board)
    center = get_board_center(board)
    size = (division_int(size[0], multiplier), division_int(size[1], multiplier))
    center = (division_int(center[0], multiplier), division_int(center[1], multiplier))
    return size, center


def get_settings(**kwargs) -> UserSettings:
    user_settings = UserSettings()
    all_flag, pow_flag, gnd_flag, all_wd, all_cl, pow_wd, pow_cl, gnd_wd, gnd_cl, keep_flag, multiple_layers = kwargs.values()
    if all_flag:
        user_settings.set_width('ALL', all_wd)
        user_settings.set_clearance('ALL', all_cl)
    if pow_flag:
        user_settings.set_width('POW', pow_wd)
        user_settings.set_clearance('POW', pow_cl)
    if gnd_flag:
        user_settings.set_width('GND', gnd_wd)
        user_settings.set_clearance('GND', gnd_cl)
    user_settings.keep = keep_flag
    user_settings.layers = multiple_layers
    return user_settings



def get_data_for_GA(filename, **kwargs):
    board = pcbnew.LoadBoard(filename)

    user_settings = get_settings(**kwargs) #
    multiplier = user_settings.get_multiplier()

    if user_settings.keep:
        existing_tracks, multiplier = get_settings_from_existing_tracks(board, user_settings)
    else:
        existing_tracks = None
    

    user_settings.change_settings(multiplier)

    size, center = get_board_dimensions(board, multiplier)
    rule_areas = get_rule_areas(board, multiplier)
    pads = get_pads_parallel(board, multiplier, 3)

    if existing_tracks:
        existing_tracks = remove_useless_routes(existing_tracks, pads)


    netcodes_dict = get_netcodes(board, user_settings, existing_tracks)
    netcodes_dict = remove_unusable_netcodes(netcodes_dict)


    offset_y, offset_x, max_y, max_x = get_board_offset(center, size, pads, rule_areas, netcodes_dict)


    update_values(pads, offset_y, offset_x, rule_areas)


    nets, routing_order = get_nets_from_netcode_dict(netcodes_dict, existing_tracks)

    update_coord_in_nets(nets, routing_order, offset_y, offset_x, multiplier)

    z = user_settings.layers
    template_grid = get_template_grid((z, max_y, max_x), rule_areas, pads)
    
    if existing_tracks:
        mark_existing_tracks_on_grid(template_grid, (z, max_y, max_x), existing_tracks, offset_y, offset_x)

    return template_grid, multiplier, offset_y, offset_x, pads, nets, routing_order, z



# testare
if __name__  == "__main__":
    board = pcbnew.LoadBoard("C:\\Users\\manue\\Desktop\\KiCAD\\2_routed\\2.kicad_pcb")
    #board = load_pcb_from_file("C:\\Users\\manue\\Desktop\\KiCAD\\test\\test.kicad_pcb")

    settings = UserSettings()
    max_unit = settings.get_multiplier()

    if True:    # keep values = True
        existing_tracks, multiplier = get_settings_from_existing_tracks(board, settings)
    else:
        existing_tracks = None
        multiplier = settings.get_multiplier()


    settings.change_settings(multiplier)

    size, center = get_board_dimensions(board, multiplier)

    rules = get_rule_areas(board, multiplier)

    pads = get_pads_parallel(board, multiplier, 3)

    if existing_tracks:
        existing_tracks = remove_useless_routes(existing_tracks, pads)

    netcodes = get_netcodes(board, settings, existing_tracks)
    netcodes = remove_unusable_netcodes(netcodes)

    offset_y, offset_x, max_y, max_x = get_board_offset(center, size, pads, rules, netcodes)

    update_values(pads = pads, offset_x = offset_x, offset_y = offset_y, rule_areas = rules)

    nets, routing_order = get_nets_from_netcode_dict(netcodes, existing_tracks)
    update_coord_in_nets(nets, routing_order, offset_y, offset_x, multiplier)


    z = settings.layers
    grid = get_template_grid((z, max_y, max_x), rules, pads)


    if existing_tracks:
        grid = mark_existing_tracks_on_grid(grid, (z, max_y, max_x), existing_tracks, offset_y, offset_x)

    paths = get_paths(grid, (z, max_y, max_x), nets, routing_order, pads, False)

    for index in range(len(paths)):
        paths[index].update_simplified_path()
        print("\n", paths[index].simplified_path, "\n")

    segments = get_segments_for_board(paths, nets, (offset_y, offset_x), multiplier)

    write_segments_to_EOF("", segments)

    see_changes()
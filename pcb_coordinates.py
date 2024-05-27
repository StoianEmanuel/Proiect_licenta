import sys
import numpy as np
sys.path.append("C:\\Program Files\\KiCad\\7.0\\bin")

import pcbnew # type: ignore
from utils import Pad
from pcb_track import get_tracks # va trebui sa imi dea clearance + width + nr de trace

def load_pcb_board(filepath):
    return pcbnew.LoadBoard(filepath)


# Conversion from nm to mm
def nm_to_mm(value):
    value /= 1e6


def get_board_center(board: pcbnew.BOARD):
    try:
        center_x, center_y = board.GetFocusPosition()
        return center_x, center_y
    except Exception as e:
        print(e)
    return 148501000, 105000000 # in nm; (297.002/2, 210.000/2) mm A4 sheet size


#
def get_board_size(board: pcbnew.BOARD):
    try: # get size using bounding box
        bounding_box = board.GetBoundingBox()
        width = bounding_box.GetWidth()
        height = bounding_box.GetHeight()
        return width, height
    except Exception as e:
        print(e)
    return 297002000, 210000000 # in nm; (297.002, 210.000) mm A4 sheet size


# Input a list of coordinates (for trace, pad) and a value reprezeting the offset that will be added / substracted
def update_coord(coord_list, offset_x, offset_y):
    updated_list = [(x - offset_x, y - offset_y) for (x, y) in coord_list]
    return updated_list


def get_shape_from_vertices(vertices_list):
    min_x = int(min(v[0] for v in vertices_list))
    max_x = int(max(v[0] for v in vertices_list))
    min_y = int(min(v[1] for v in vertices_list))
    max_y = int(max(v[1] for v in vertices_list))

    coord_list = []

    # Functie pentru a verifica daca un punct este in interiorul poligonului
    def is_point_in_polygon(x, y, vertices):    # ray-casting
        n = len(vertices)
        inside = False
        px, py = x, y
        for i in range(n):
            j = (i + 1) % n
            vi = vertices[i]
            vj = vertices[j]
            if ((vi[1] > py) != (vj[1] > py)) and \
                (px < (vj[0] - vi[0]) * (py - vi[1]) / (vj[1] - vi[1]) + vi[0]):
                inside = not inside
        return inside

    # Generare de coordonate
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if is_point_in_polygon(x, y, vertices_list):
                coord_list.append((x, y))

    return coord_list


def scale_coordinates(vertex, scale_factor):    # (x,y), z
    """
    Scale the coordinates of a vertex using the given scale factor.
    """
    scaled_x, scaled_y = vertex
    scaled_x /= scale_factor
    scaled_y /= scale_factor
    return scaled_x, scaled_y



def process_polygon(polygon, scale_factor):
    """
    Process the vertices of a polygon and scale them using the given scale factor.
    """
    vertices_list = []
    for i in range(polygon.VertexCount()):
        vertex = polygon.CVertex(i)
        scaled_vertex = scale_coordinates(vertex, scale_factor)
        vertices_list.append(scaled_vertex)
    return vertices_list



def get_rule_areas(board: pcbnew.BOARD, scale_factor: int = 1):
    areas = []
    for zone in board.Zones():
        if zone.GetLayer() == pcbnew.F_Cu:
            polygon = zone.Outline()
            vertices_list = process_polygon(polygon, scale_factor)
            rule_area = get_shape_from_vertices(vertices_list)
            areas.append(rule_area)
    return areas


def get_pads(board: pcbnew.BOARD, scale_factor: int = 1):
    pads = []
    footprints = board.GetFootprints()
    for part in footprints:
        for pad in part.Pads():
            center_x, center_y = pad.GetPosition()
            length, width = pad.GetSize()
            angle = pad.GetOrientationDegrees()
            polygon = pad.GetEffectivePolygon()
            vertices_list = process_polygon(polygon, scale_factor)
            occupied_area = get_shape_from_vertices(vertices_list=vertices_list)
            aux = Pad(center_x = center_x // scale_factor, center_y = center_y // scale_factor,
                      original_center_x = center_x, original_center_y = center_y,
                      length = length, width = width,
                      angle = angle, occupied_area = occupied_area,
                      pad_name = pad.GetName(), part_name = None)
            pads.append(aux)
    return pads


def get_scale(value, unit = 10):
    factor = 1
    while value != int(value):
        value *= 10
        factor *= unit
    return factor


def get_interval(area, current_max_x, current_max_y, current_min_x, current_min_y, scale): # area = list of tuple coord
    aux = np.array(area)
    min_x = np.min(aux[:, 0])
    min_y = np.min(aux[:, 1])
    max_x = np.max(aux[:, 0])
    max_y = np.max(aux[:, 1])
    min_x = min(min_x, current_min_x)
    min_y = min(min_y, current_min_y)
    max_x = max(max_x, current_max_x)
    max_y = max(max_y, current_max_y)

    for (x, y) in area:
        scale = max(scale, get_scale(x, 10))
        scale = max(scale, get_scale(y, 10))
    
    return current_max_x, current_max_y, current_min_x, current_min_y


# Offset for x and y -- hepls reducing grid size
def get_offset_and_scale(board: pcbnew.BOARD):
    center_x, center_y = get_board_center(board)
    width, length = get_board_size(board)
    origin_x, origin_y = center_x - width // 2, center_y - length // 2
    end_x, end_y = center_x + width // 2, center_y + length // 2

    pads = get_pads(board = board, scale_factor = 1)
    rule_areas = get_rule_areas(board = board, scale_factor = 1)

    max_x = float('-inf')
    max_y = float('-inf')
    min_x = float('inf')
    min_y = float('inf')
    scale = float('-inf')
    if rule_areas:
        for area in rule_areas:
            if area:
                max_x, max_y, min_x, min_y = get_interval(area, max_x, max_y, min_x, min_y, scale)

    for pad in pads:
        area = pad.occupied_area
        max_x, max_y, min_x, min_y = get_interval(area, max_x, max_y, min_x, min_y, scale)


    # se va tine cont si de nr de trace-uri (width + clerance)
    offset_top_x = max(0, min_x - origin_x)
    offset_top_y = max(0, min_y - origin_y)
    offset_bottom_x = min(0, end_x - max_x)
    offset_bottom_y = min(0, end_y - max_y)
    return offset_top_x, offset_top_y, offset_bottom_x, offset_bottom_y, scale


def update_values():
    ...


def generate_grid():
    ...

import sys
sys.path.append("C:\\Program Files\\KiCad\\7.0\\bin")

import pcbnew

board = pcbnew.LoadBoard("C:\\Users\\manue\\Desktop\\KiCAD\\test\\test.kicad_pcb")

# ----- dimensiuni
#pcb_coordinates.py


# -------------- routed / unrouted


print("---------------")

# Obține toate neturile și coordonatele capetelor lor pentru neturile nerutate
unrouted_nets = {}
pcb = board
for track in pcb.GetTracks():
    if isinstance(track, pcbnew.PCB_TRACK) and not track.IsSelected():
        net_code = track.GetNetCode()
        net = pcb.FindNet(net_code)
        if net:
            net_name = net.GetNetname()
            if net_name not in unrouted_nets:
                unrouted_nets[net_name] = []
            start_x, start_y = pcbnew.ToMM(track.GetStart().x), pcbnew.ToMM(track.GetStart().y)
            end_x, end_y = pcbnew.ToMM(track.GetEnd().x), pcbnew.ToMM(track.GetEnd().y)
            unrouted_nets[net_name].append(((start_x, start_y), (end_x, end_y)))

# Eliminăm neturile duplicate
unique_unrouted_nets = {net_name: coords for net_name, coords in unrouted_nets.items() if len(coords) == 1}

# Obținem informațiile despre pad-urile asociate fiecărui net
for net_name, coords in unique_unrouted_nets.items():
    print(f"Net: {net_name}")
    for coord_pair in coords:
        (start_x, start_y), (end_x, end_y) = coord_pair
        print(f"   Coordinates: ({start_x}, {start_y}) - ({end_x}, {end_y})")
        # Aici puteți obține informațiile despre pad-urile asociate netului și să le procesați cum doriți
        # Exemplu: pentru fiecare pad asociat netului, obțineți rotirea, forma și centrul pad-ului


# -----------

# Funcție pentru conversia din nm în mm
def nm_to_mm(value_nm):
    return value_nm / 1e6

# Listă pentru a stoca informațiile despre trasee
tracks_info = []

# Iterăm prin toate traseele și extragem informațiile relevante
for track in pcb.GetTracks():
    if isinstance(track, pcbnew.PCB_TRACK):
        start = track.GetStart()
        end = track.GetEnd()
        width = track.GetWidth()
        netname = track.GetNetname()
        clearance = track.GetLocalClearance(start)

        track_info = {
            "netname": netname,
            "start": (nm_to_mm(start.x), nm_to_mm(start.y)),
            "end": (nm_to_mm(end.x), nm_to_mm(end.y)),
            "width": nm_to_mm(width),
            "clearance": nm_to_mm(clearance)
        }

        tracks_info.append(track_info)

# Afișează informațiile despre trasee
for info in tracks_info:
    print(f"Net: {info['netname']}, Start: {info['start']}, End: {info['end']}, Width: {info['width']} mm, Clearance: {info['clearance']} mm")


# -------

# Funcție pentru obținerea vârfurilor unui poligon
def get_polygon_vertices(pad):
    vertices = []
    polygon = pad.GetEffectivePolygon()
    for i in range(polygon.VertexCount()):
        vertex = polygon.CVertex(i)
        vertices.append((nm_to_mm(vertex.x), nm_to_mm(vertex.y)))
    return vertices

# Listă pentru a stoca informațiile despre pad-uri și vârfurile lor
pads_info = []

# Iterăm prin toate modulele și pad-urile lor și extragem informațiile relevante
modules = pcb.GetFootprints()
for module in modules:
    for pad in module.Pads():
        pad_info = {
            "name": pad.GetName(),
            "netname": pad.GetNet().GetNetname(),
            "vertices": get_polygon_vertices(pad)
        }
        pads_info.append(pad_info)

# Afișează informațiile despre pad-uri și vârfurile poligoanelor
for info in pads_info:
    print(f"Pad: {info['name']}, Net: {info['netname']}, Vertices: {info['vertices']}")

# Funcție pentru a crea o matrice care să reprezinte pad-ul
def create_pad_matrix(vertices):
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)

    # Dimensiunea matricei
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)
    matrix = [[0 for _ in range(width)] for _ in range(height)]

    # Functie pentru a verifica daca un punct este in interiorul poligonului
    def is_point_in_polygon(x, y, vertices):
        n = len(vertices)
        inside = False
        px, py = x, y
        for i in range(n):
            j = (i + 1) % n
            vi = vertices[i]
            vj = vertices[j]
            if ((vi[1] > py) != (vj[1] > py)) and (px < (vj[0] - vi[0]) * (py - vi[1]) / (vj[1] - vi[1]) + vi[0]):
                inside = not inside
        return inside

    # Marcare matrice
    for y in range(height):
        for x in range(width):
            if is_point_in_polygon(min_x + x, min_y + y, vertices):
                matrix[y][x] = 1

    return matrix

# Crearea matricii pentru fiecare pad
for info in pads_info:
    matrix = create_pad_matrix(info["vertices"])
    print(f"Pad: {info['name']}, Matrix:")
    for row in matrix:
        print(' '.join(map(str, row)))


import math

# Funcție pentru determinarea factorului de scalare
def determine_scale_factor(values):
    max_decimal_places = 0
    for value in values:
        str_value = str(value)
        if '.' in str_value:
            decimal_places = len(str_value.split('.')[1])
            max_decimal_places = max(max_decimal_places, decimal_places)
    return 10 ** max_decimal_places

# Funcție pentru scalare
def scale_values(values, scale_factor):
    return [value * scale_factor for value in values]

# Funcție pentru obținerea coordonatelor vârfurilor poligonului unui pad
def get_polygon_vertices(pad):
    vertices = []
    polygon = pad.GetEffectivePolygon()
    for i in range(polygon.VertexCount()):
        vertex = polygon.CVertex(i)
        vertices.append((nm_to_mm(vertex.x), nm_to_mm(vertex.y)))
    return vertices

# Extrage informațiile despre pad-uri
pads_info = []

modules = pcb.GetFootprints()
for module in modules:
    for pad in module.Pads():
        pad_info = {
            "name": pad.GetName(),
            "netname": pad.GetNet().GetNetname(),
            "width": nm_to_mm(pad.GetSize().x),
            "length": nm_to_mm(pad.GetSize().y),
            "position": (nm_to_mm(pad.GetPosition().x), nm_to_mm(pad.GetPosition().y)),
            "vertices": get_polygon_vertices(pad)
        }
        pads_info.append(pad_info)

# Determină factorul de scalare
all_values = []
for info in pads_info:
    all_values.append(info["width"])
    all_values.append(info["length"])
    all_values.extend(info["position"])
    for vertex in info["vertices"]:
        all_values.extend(vertex)

scale_factor = determine_scale_factor(all_values)

# Scalare a valorilor
for info in pads_info:
    info["width"] = int(info["width"] * scale_factor)
    info["length"] = int(info["length"] * scale_factor)
    info["position"] = (int(info["position"][0] * scale_factor), int(info["position"][1] * scale_factor))
    info["vertices"] = [(int(vertex[0] * scale_factor), int(vertex[1] * scale_factor)) for vertex in info["vertices"]]

# Afișează informațiile scalate despre pad-uri
for info in pads_info:
    print(f"Pad: {info['name']}, Net: {info['netname']}, Width: {info['width']}, Length: {info['length']}, Position: {info['position']}, Vertices: {info['vertices']}")

# (Opțional) Funcție pentru a reduce dimensiunile cu cmmdc
def gcd_multiple(numbers):
    return math.gcd(*numbers)

# (Opțional) Aplicare cmmdc
widths = [info["width"] for info in pads_info]
lengths = [info["length"] for info in pads_info]
positions = [coord for info in pads_info for coord in info["position"]]
vertices = [coord for info in pads_info for vertex in info["vertices"] for coord in vertex]

all_scaled_values = widths + lengths + positions + vertices
common_divisor = gcd_multiple(all_scaled_values)

for info in pads_info:
    info["width"] //= common_divisor
    info["length"] //= common_divisor
    info["position"] = (info["position"][0] // common_divisor, info["position"][1] // common_divisor)
    info["vertices"] = [(vertex[0] // common_divisor, vertex[1] // common_divisor) for vertex in info["vertices"]]

# Afișează informațiile scalate și reduse despre pad-uri
for info in pads_info:
    print(f"Pad: {info['name']}, Net: {info['netname']}, Width: {info['width']}, Length: {info['length']}, Position: {info['position']}, Vertices: {info['vertices']}")

import sys
sys.path.append("C:\\Program Files\\KiCad\\7.0\\bin")

import pcbnew # type: ignore


def get_tracks(pcb: pcbnew.BOARD):
    aux = pcb.GetTracks()
    return aux


def get_default_width(pcb: pcbnew.BOARD):
    design_settings = pcb.GetDesignSettings()
    default_clearance = design_settings.m_TrackMinWidth  # nm
    return default_clearance


def get_min_width(pcb: pcbnew.BOARD):   # Used in case that no width is specified or there is no default width 
    min_trace_width = float("inf")
    for item in pcb.GetTracks():
        if isinstance(item, pcbnew.PCB_TRACK):
            track_width = item.GetWidth()
            if track_width < min_trace_width:
                min_trace_width = track_width
    return min_trace_width

# O voi evita ------------------------------
def get_min_clearance(pcb: pcbnew.BOARD):
    min_clearance = float("inf")
    for item in pcb.GetTracks():
        if isinstance(item, pcbnew.ZONE_CONTAINER):
            clearance = item.GetClearance()
            if clearance < min_clearance:
                min_clearance = clearance
    return min_clearance

# S-ar putea sa o evit si pe aceasta ????
def get_default_clearance(pcb: pcbnew.BOARD):
    design_settings = pcb.GetDesignSettings()
    default_trace_width = design_settings.m_MinClearance # nm
    return default_trace_width


# Obține toate neturile
def get_tracks(pcb: pcbnew.BOARD):
    # Creează un dicționar pentru a stoca starea rutării fiecărui net
    net_status = {}

    # Obține toate neturile
    for netcode in range(pcb.GetNetCount()):
        net = pcb.GetNetInfo().GetNetItem(netcode)
        net_name = net.GetNetname()
        net_status[net_name] = "No_track_assigned"

    # Verifică segmentele de traseu
    for track in pcb.GetTracks():
        if isinstance(track, pcbnew.PCB_TRACK):
            net_code = track.GetNetCode()
            net = pcb.FindNet(net_code)
            if net:
                net_name = net.GetNetname()
                # Dacă există cel puțin un segment de traseu pentru un net, îl marcăm ca rutat
                net_status[net_name] = "Needs_track"

    # Afișează starea fiecărui net
    for net, status in net_status.items():
        print(f"Net: {net}, Status: {status}")


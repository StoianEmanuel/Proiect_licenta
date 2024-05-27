import numpy as np
import matplotlib.pyplot as plt # ----
import copy

# COLORS used for different paths, in final form all routes same color
COLORS = {  'white' : [1,1,1],          'black' : [0.0,0.0,0.0],    'red'   : [1,0.0,0.0],  'green' : [0.0,1,0.0],
            'orange': [1,1,0.0],        'blue'  : [0.0,0.0,1],      'yellow': [0.0,0.8,1],  'purple': [1,0.0,1],
            'pink'  : [0.8,0.0,0.0],    'aqua'  : [0.0,1,1]
}



# return an array filled with background color and colors assigned to pins 
def get_RGB_matrix(nodes, colors_list, rows: int, columns: int, background = COLORS['white']):
    matrix = [[background for _ in range(columns)] for _ in range(rows)] # used to assign colors for routes

    for i in range(len(colors_list)):
        x = np.array(nodes).shape
        if x != (4,): #  at least 2 routes
            pin1_x, pin1_y, pin2_x, pin2_y = nodes[i] # nodes coordsd
        else:
            pin1_x = nodes[0];  pin1_y = nodes[1]
            pin2_x = nodes[2];  pin2_y = nodes[3]

        matrix[pin1_x][pin1_y] = colors_list[i]
        matrix[pin2_x][pin2_y] = colors_list[i]
    
    return matrix


def color_pads_in_RGB_matrix(pads, rows: int, columns: int, color = COLORS['aqua'], background = COLORS['white'], grid = None):
    if grid:
        matrix = copy.deepcopy(grid)
    else:
        matrix = [[background for _ in range(columns)] for _ in range(rows)] # used to assign colors for routes

    for pad in pads:
        area = pad.occupied_area
        for coord_set in area:
            x, y = coord_set
            matrix[x][y] = color
    
    return matrix


# Draw the grid and update color_matrix with the latest path
def draw_grid(color_matrix, main_path, color_main_path = COLORS["yellow"], other_nodes = None, color_other_nodes = COLORS['orange']):    # 2024-03-13 16:00:53
    if main_path != None and len(main_path) > 0:
        for i in main_path: # assign color to path
            x = i[0]
            y = i[1]
            color_matrix[x][y] = color_main_path 
        
        x, y = main_path[0]
        color_matrix[x][y] = COLORS["aqua"]    # color assignement for pins
        x, y = main_path[-1]
        color_matrix[x][y] = COLORS["aqua"]

    if other_nodes != None and len(other_nodes) > 0:
        for i in other_nodes: # assign color to path
            x = i[0]
            y = i[1]
            color_matrix[x][y] = color_other_nodes 
        

    arr = np.array(color_matrix, dtype=float)

    plt.imshow(arr, origin='upper', extent=[0.0, 1, 0.0, 1])
    plt.axis("off")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# COLORS used for different paths, in final form all routes same color
COLORS = {  'white' : [1,1,1],          'black' : [0.0,0.0,0.0],    'red'   : [1,0.0,0.0],  'green' : [0.0,1,0.0],
            'orange': [1,1,0.0],        'blue'  : [0.0,0.0,1],      'yellow': [0.0,0.8,1],  'purple': [1,0.0,1],
            'pink'  : [0.8,0.0,0.0],    'aqua'  : [0.0,1,1]
}

ROWS = 100
COLS = 100



# return an array filled with background color and colors assigned to pins 
def get_RGB_matrix(nodes, colors_list, background = COLORS['white'], rows = ROWS, columns = COLS):
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



# Draw the grid and update color_matrix with the latest path
def draw_grid(color_matrix, path, color = COLORS["yellow"]):    # 2024-03-13 16:00:53
    if path != None:
        for i in path: # assign color to path
            x = i[0]
            y = i[1]
            color_matrix[x][y] = color 
        
        x, y = path[0]
        color_matrix[x][y] = COLORS["aqua"]    # color assignement for pins
        x, y = path[-1]
        color_matrix[x][y] = COLORS["aqua"]

    arr = np.array(color_matrix, dtype=float)

    plt.imshow(arr, origin='upper', extent=[0.0, 1, 0.0, 1])
    plt.axis("off")
    plt.show()

from matplotlib import pyplot as plt

# COLORS used for different paths, in final form all routes same color
COLORS = {  'white' : [1,1,1],          'black' : [0.0,0.0,0.0],    'red'   : [1,0.0,0.0],  'green' : [0.0,1,0.0],
            'orange': [1,1,0.0],        'blue'  : [0.0,0.0,1],      'yellow': [0.0,0.8,1],  'purple': [1,0.0,1],
            'pink'  : [0.8,0.0,0.0],    'aqua'  : [0.0,1,1]
}

# Trace the path from source to destination
def trace_path(cell_details, dest, color_matrix, color = COLORS["yellow"], value = 1):
    message = f"Path {value}. is"
    print(message)
    path = []
    row, col = dest.x, dest.y

    # Trace path from dest to start using parent cells
    while not (cell_details[row][col].parent.x == row and cell_details[row][col].parent.y == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent.x
        temp_col = cell_details[row][col].parent.y
        row = temp_row
        col = temp_col

    path.append((row, col))
    path.reverse()

    for i in path:  print("->", i, end="")
    
    for i in path: # assign color to path
        x = i[0]
        y = i[1]
        color_matrix[x][y] = color 

    color_matrix[row][col] = COLORS["green"]    # color assignement for pins
    color_matrix[path[len(path)-1][0]][path[(len(path)-1)][1]] = COLORS["green"]

    plt.imshow(color_matrix)
    plt.axis("off")
    plt.show()
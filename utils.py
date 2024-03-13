import numpy as np
from math import sqrt

'''
depot_x = 4             # location of depot
depot_y = 3
x = np.array([0,3,6,7,15,10,16,5,8,1.5])        # customers locations for printing on screen
y = np.array([1,2,1,4.5,-1,2.5,11,6,9,12])
'''

def matrice_adiacenta(depot_x, depot_y, x, y):
    size = len(x)
    matr = np.zeros((size,size), dtype=float)
    for i in range(size - 1):
        for j in range(size - 1):
            if i!=j:
                matr[i+1][j+1] = sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2) 

    for i in range(size - 1):
        matr[0][i+1] = sqrt((x[i] - depot_x)**2 + (y[i] - depot_y)**2) 
        matr[i+1][0] = sqrt((x[i] - depot_x)**2 + (y[i] - depot_y)**2) 

    return matr



'''movement heuristics types'''
# 4 directions
def h_manhattan(row: int, col: int, dest):
    return abs(row - dest.x) + abs(col - dest.y)

# any direction
def h_euclidian(row: int, col: int, dest):
    return sqrt((row - dest.x)**2 + (col - dest.y)**2)

# 8 directions
def h_diagonal(row: int, col: int, dest):
    dx = abs(row - dest.x)
    dy = abs(col - dest.y)
    D  = 1  # node length
    D2 = 1.41421 #sqrt(2) - diagonal distance between nodes
    return D * (dx + dy) + (D2 - 2*D) * min(dx, dy)
''''''


# return a rectangle area of cells
def generate_rectangle(row: int, col: int, length_x: int, length_y):
    area = [(i+row, j+col) for j in range(length_y) for i in range(length_x)]
    return area

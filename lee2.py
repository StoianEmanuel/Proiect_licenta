# pe baza de cozi - inefiecient dpdv al timpului si al matrcilor folosite 

import csv
from collections import deque
import time

ROWS = 100
COLS = 100

# Clasa folosita pentru a retine coordonatele celulelor/punctelor
class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


# Folosita pentru BFS
class queueNode:
    def __init__(self, point: Cell, dist: int):
        self.point = point
        self.dist  = dist    # Distanta punct fata de sursa


# Verifica daca celula se afla intr-o anumita regiune (in spatiul [0 - (COLS-1)] x [0 - (ROWS-1)]
def checkValid(row: int, col: int):
    return ((row >= 0 and row < ROWS) and (col >= 0 and col < COLS))


# Mutarile disponibile
movVert  = [-1, 0, 1, 0]
movHoriz = [0, -1, 0, 1]


# Lee Routing pentru gasirea unei rute de lungime minima
def bfsLee(matr, src: Cell, dest: Cell):

    # Verificare daca src sau dest sunt marcate in matrice
    if matr[src.x][src.y] != 1 or matr[dest.x][dest.y] != 1:
        return -1, None, 0
    
    # Pentru a verifica daca nodul/punctul a fost parcurs
    visited = [[False for i in range(COLS)] for j in range(ROWS)]
    
    visited[src.x][src.y] = True

    # Crearea unei cozi pentru a retine drumul
    q = deque()
    s = queueNode(src, 0)   # definirea si adaugarea nodului sursa in coada
    q.append(s)

    # BFS pornind de la nodul sursa
    while q:
        curr = q.pop()  # Extragerea primului element cel mai din stanga / inceput

        point = curr.point
        if point.x == dest.x and point.y == dest.y:
            return curr.dist, visited, len(q)    # Daca s-a ajuns la destinatie returneaza distanta drumului
        
        for i in range(4):
            row = point.x + movVert[i]
            col = point.y + movHoriz[i]

            if checkValid(row, col) and matr[row][col] and not visited[row][col]: # matr[row][col] != val -- pentru a ma asigura ca nu exista obstecole
                visited[row][col] = True
                auxCell = Cell(row, col)
                auxNode = queueNode(auxCell, curr.dist + 1)
                q.append(auxNode)

    return -1, None, 0   # Daca nu exista drum intre sursa si destinatie


matr = [[1 for i in range(COLS)] for j in range(ROWS)]

src  = Cell(20, 10)
dest = Cell(30, 80)

start = time.perf_counter()

dist, visited, queueSize = bfsLee(matr, src, dest)

stop = time.perf_counter()

if dist != -1:
		print("Length of the Shortest Path is", dist, "in ", stop-start, "s with", queueSize)
else:
		print("Shortest Path doesn't exist", "in ", stop-start, "s")


# Exportarea matricei în fișierul CSV
cale_fisier_csv = 'matrice_Lee2.csv'
with open(cale_fisier_csv, 'w', newline='') as fisier_csv:
    writer = csv.writer(fisier_csv)
    writer.writerows(visited)
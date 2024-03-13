# implementare fara clasa, rudimentara, care necesita optimizari

import numpy as np
from itertools import product
import csv
import time

size = 100
matr = np.zeros((size, size), dtype=int)

startX, startY = 10, 20
stopX, stopY = 80, 30

ok = False
i = 0

matr[startY][startX] = -1
matr[stopY][stopX]   = -1

path = []


# verifica pe baza de margini de patrate / dreptunghiuri daca destinatie se afla pe acea muchie si completez matricea cu valori
# pentru a putea reconstitui drumul
while not ok:
    i += 1
    if i > size:
        break

    minX = startX - i
    if minX >= 0:
        minY = max(0, startY - i)
        maxY = min(size - 1, startY + i)
        for k in range(minY, maxY + 1):
            if matr[k][minX] != -1:
                matr[k][minX] = i
            else:
                ok = True
                break
                
    
    maxX = startX + i
    if maxX < size:
        minY = max(0, startY - i)
        maxY = min(size - 1, startY + i)
        for k in range(minY, maxY + 1):
            if matr[k][maxX] != -1:
                matr[k][maxX] = i
            else:
                ok = True
                break


    minY = startY - i
    if minY >= 0:
        minX = max(0, startX - i)
        maxX = min(size - 1, startX + i)
        for k in range(minX, maxX + 1):
            if matr[minY][k] != -1:
                matr[minY][k] = i
            else:
                ok = True
                break
    

    maxY = startY + i
    if maxY < size:
        minX = max(0, startX - i)
        maxX = min(size - 1, startX + i)
        for k in range(minX, maxX + 1):
            if matr[maxY][k] != -1:
                matr[maxY][k] = i
            else:
                ok = True
                break


# daca am gasit drum / coordonata: incep de la final catre start prin maximul din jur si verific daca se gaseste valoarea negativa
# printre vecinii celulei

#for t in range(size):
#    print(matr[t])
            

# Exportarea matricei în fișierul CSV
cale_fisier_csv = 'matrice_Lee1.csv'
with open(cale_fisier_csv, 'w', newline='') as fisier_csv:
    writer = csv.writer(fisier_csv)
    writer.writerows(matr)


# Reface drum       
minX = max(0, stopX - 1)
maxX = min(size - 1, stopX + 1)
minY = max(0, stopY - 1)
maxY = min(size - 1, stopY + 1)

start = time.perf_counter()
if ok:
    print("Searching for path!")

    while ok:
        intervalX = range(minX, maxX + 1)
        intervalY = range(minY, maxY + 1)
        minimX = minX
        minimY = minY
        minim  = matr[minY][minX]

        for x, y in product(intervalX, intervalY):
            if matr[y][x] == -1: 
                if x == startX and y == startY:
                    ok = False
                    break
            elif matr[y][x] < minim and matr[y][x] != 0:
                minim = matr[y][x]
                minimX = x
                minimY = y
        
        if ok:
            path.append([minimX, minimY])

        minX = max(0, minimX - 1)
        maxX = min(size - 1, minimX + 1)
        minY = max(0, minimY - 1)
        maxY = min(size - 1, minimY + 1)
stop = time.perf_counter()

print(path)

print("Duration: ", stop-start, "s")
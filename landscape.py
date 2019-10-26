import numpy as np
from math import sin
from mathematics import *
def generateMap(scale):
    n = 30
    m = 100
    Map = np.empty((n, m, 4), dtype=np.float, order='C')
    for x in range(n):
        for y in range(m):
            Map[x][y][0] = x * scale
            Map[x][y][1] = round(sin(y / 4) * -sin(x / 8) * 8, 2) * scale
            Map[x][y][2] = y * scale
            Map[x][y][3] = 1 * scale
    return Map


def to2D(Map):
    x = Map.shape[0]
    y = Map.shape[1]
    ans = np.empty((4, x*y), dtype=np.float, order='C')
    for i in range(x):
        for j in range(y):
            for k in range(4):
                ans[k][i*y+j] = Map[i][j][k]
    return ans





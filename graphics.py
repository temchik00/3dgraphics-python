from numba import jit, uint32, void, int32, njit, prange, float32, float64
import numpy as np
from random import random


@jit(uint32(uint32, uint32, uint32), nopython=True)
def rgbToHexDecimal(r, g, b):
    res = 0
    res += (r // 16) * 1048576
    res += (r % 16) * 65536

    res += (g // 16) * 4096
    res += (g % 16) * 256

    res += int(b // 16) * 16
    res += b % 16

    return res


@jit(float32[:](int32[:, :], int32, int32), nopython=True)
def barycentric(triangle, x, y):
    u = np.cross(
        np.asarray((triangle[2][0] - triangle[0][0], triangle[1][0] - triangle[0][0], triangle[0][0] - x)),
        np.asarray((triangle[2][1] - triangle[0][1], triangle[1][1] - triangle[0][1], triangle[0][1] - y)))
    if abs(u[2]) < 1:
        return np.asarray((-1, 1, 1), dtype=float32)
    return np.asarray((1.0 - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2]), dtype=float32)


@njit(void(uint32[:, :], int32[:], int32[:], uint32, uint32, uint32), parallel=True)
def drawLine(surface, point0, point1, r, g, b):
    a = np.subtract(point0, point1)
    dx = abs(a[0])
    dy = abs(a[1])
    if dx >= dy:
        if point0[0] > point1[0]:
            point0, point1 = point1, point0
        if a[1] == 0:
            for x in prange(point0[0], point1[0]):
                if surface[x][point0[1]] == 16777215:
                    surface[x][point0[1]] = rgbToHexDecimal(r, g, b)
        else:
            k = a[1] / a[0]
            b = point0[1] - point0[0] * k
            for x in prange(point0[0], point1[0]):
                y = round(k * x + b)
                if surface[x][int(y)] == 16777215:
                    surface[x][int(y)] = rgbToHexDecimal(r, g, b)
    else:
        if point0[1] > point1[1]:
            point0, point1 = point1, point0
        if a[0] == 0:
            for y in prange(point0[1], point1[1]):
                if surface[point0[0]][y] == 16777215:
                    surface[point0[0]][y] = rgbToHexDecimal(r, g, b)
        else:
            k = a[0] / a[1]
            b = point0[0] - point0[1] * k
            for y in prange(point0[1], point1[1]):
                x = round(k * y + b)
                if surface[int(x)][y] == 16777215:
                    surface[int(x)][y] = rgbToHexDecimal(r, g, b)
    return


@njit(void(uint32[:], uint32[:, :], int32[:, :]), parallel=True)
def drawTriangle(screenSize, surface, triangle):
    # if triangle[0][2] < 0.05 or triangle[1][2] < 0.05 or triangle[2][2] < 0.05:
    #     return
    boxMin = np.empty(2, dtype=np.int32)
    boxMax = np.empty(2, dtype=np.int32)
    boxMin[0] = min(triangle[0][0], triangle[1][0], triangle[2][0])
    if boxMin[0] < 0:
        boxMin[0] = 0
    elif boxMin[0] >= screenSize[0]:
        return
    boxMin[1] = min(triangle[0][1], triangle[1][1], triangle[2][1])
    if boxMin[1] < 0:
        boxMin[1] = 0
    elif boxMin[1] >= screenSize[1]:
        return
    boxMax[0] = max(triangle[0][0], triangle[1][0], triangle[2][0])
    if boxMax[0] >= screenSize[0]:
        boxMax[0] = screenSize[0]-1
    if boxMax[0] < 0:
        return
    boxMax[1] = max(triangle[0][1], triangle[1][1], triangle[2][1])
    if boxMax[1] >= screenSize[1]:
        boxMax[1] = screenSize[1]-1
    if boxMax[1] < 0:
        return
    color = random() * 1000000
    for x in prange(boxMin[0], boxMax[0]+1):
        for y in prange(boxMin[1], boxMax[1] + 1):
            if surface[x][y] == 16777215:
                u = barycentric(triangle, x, y)
                if u[0] >= 0 and u[1] >= 0 and u[2] >= 0:
                    surface[x][y] = color


@njit(void(uint32[:], uint32[:, :], float64[:, :], int32, int32), parallel=True)
def drawTriangles(screenSize, surface, triangles, n, m):
    for i in range(n-1):
        for j in range(m-1):
            if triangles[(i + 1) * m + j][3] > 0.01 and triangles[i * m + j + 1][3] > 0.01:
                triangle = np.empty((3, 3), dtype=np.int32)
                for k in prange(3):
                    triangle[1][k] = round(triangles[(i + 1) * m + j][k] / triangles[(i + 1) * m + j][3])
                    triangle[2][k] = round(triangles[i * m + j + 1][k] / triangles[i * m + j + 1][3])
                if triangles[i * m + j][3] > 0.01:
                    for k in range(3):
                        triangle[0][k] = round(triangles[i * m + j][k] / triangles[i * m + j][3])
                    drawTriangle(screenSize, surface, triangle)
                if triangles[(i + 1) * m + j + 1][3] > 0.01:
                    for k in range(3):
                        triangle[0][k] = round(triangles[(i + 1) * m + j + 1][k] / triangles[(i + 1) * m + j + 1][3])
                    drawTriangle(screenSize, surface, triangle)


@njit(void(uint32[:, :], uint32), parallel=True)
def clearScreen(screen, colorHex):
    for i in prange(screen.shape[0]):
        for j in prange(screen.shape[1]):
            screen[i][j] = colorHex


@njit(void(int32[:, :], int32), parallel=True)
def clearBuffer(zBuffer, clipDist):
    for i in prange(zBuffer.shape[0]):
        for j in prange(zBuffer.shape[1]):
            zBuffer[i][j] = clipDist
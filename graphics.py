from numba import jit, uint64, void, int64, njit, prange, float64, float32, int32, uint32
import numpy as np
from random import random


@jit(uint64(uint64, uint64, uint64), nopython=True)
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
    if u[2] == 0:
        return np.asarray((-1, 1, 1), dtype=float32)
    return np.asarray((1.0 - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2]), dtype=float32)


@njit(void(uint64[:], uint32[:, :], int32[:, :], uint32, float32[:, :]), parallel=True)
def drawTriangle(screenSize, surface, triangle, color, zbuffer):
    boxMin = np.empty(2, dtype=np.int64)
    boxMax = np.empty(2, dtype=np.int64)
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

    for x in prange(boxMin[0], boxMax[0]+1):
        for y in prange(boxMin[1], boxMax[1] + 1):
            u = barycentric(triangle, x, y)
            z = triangle[0][2] * u[0] + triangle[1][2] * u[1] + triangle[2][2] * u[2]
            if u[0] >= 0 and z < zbuffer[x][y] and u[1] >= 0 and u[2] >= 0:
                zbuffer[x][y] = z
                surface[x][y] = color


@njit(void(uint64[:], uint32[:, :], float32[:, :], int64[:, :], float32[:, :], float64))
def drawPolys(screenSize, surface, points, faces, zbuffer, depth):
    for face in range(faces.shape[0]):
        if 0 < points[faces[face][0]][2] <= depth:
            color = uint32(random() * 1000000)
            triangle = np.empty((3, 3), dtype=np.int32)
            for i in range(3):
                triangle[0][i] = points[faces[face][0]][i]
            for point in range(2, faces.shape[1]):
                if faces[face][point] < 0:
                    break
                for i in range(3):
                    triangle[1][i] = points[faces[face][point - 1]][i]
                    triangle[2][i] = points[faces[face][point]][i]
                if 0 < triangle[1][2] <= depth and 0 < triangle[2][2] <= depth:
                    drawTriangle(screenSize, surface, triangle, color, zbuffer)


@njit(void(uint32[:, :], uint32), parallel=True)
def clearScreen(screen, colorHex):
    for i in prange(screen.shape[0]):
        for j in prange(screen.shape[1]):
            screen[i][j] = colorHex


@njit(void(float32[:, :], float32), parallel=True)
def clearBuffer(zBuffer, clipDist):
    for i in prange(zBuffer.shape[0]):
        for j in prange(zBuffer.shape[1]):
            zBuffer[i][j] = clipDist

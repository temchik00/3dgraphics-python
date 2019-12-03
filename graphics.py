from numba import jit, uint64, void, int64, njit, prange, float64, float32, int32, uint32
import numpy as np
from random import random
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

bdim = (8, 8, 1)


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


# @njit(void(uint64[:, :], int64[:], int64[:], uint64, uint64, uint64), parallel=True)
# def drawLine(surface, point0, point1, r, g, b):
#     a = np.subtract(point0, point1)
#     dx = abs(a[0])
#     dy = abs(a[1])
#     if dx >= dy:
#         if point0[0] > point1[0]:
#             point0, point1 = point1, point0
#         if a[1] == 0:
#             for x in prange(point0[0], point1[0]):
#                 if surface[x][point0[1]] == 16777215:
#                     surface[x][point0[1]] = rgbToHexDecimal(r, g, b)
#         else:
#             k = a[1] / a[0]
#             b = point0[1] - point0[0] * k
#             for x in prange(point0[0], point1[0]):
#                 y = round(k * x + b)
#                 if surface[x][int(y)] == 16777215:
#                     surface[x][int(y)] = rgbToHexDecimal(r, g, b)
#     else:
#         if point0[1] > point1[1]:
#             point0, point1 = point1, point0
#         if a[0] == 0:
#             for y in prange(point0[1], point1[1]):
#                 if surface[point0[0]][y] == 16777215:
#                     surface[point0[0]][y] = rgbToHexDecimal(r, g, b)
#         else:
#             k = a[0] / a[1]
#             b = point0[0] - point0[1] * k
#             for y in prange(point0[1], point1[1]):
#                 x = round(k * y + b)
#                 if surface[int(x)][y] == 16777215:
#                     surface[int(x)][y] = rgbToHexDecimal(r, g, b)
#     return

mod = SourceModule("""
__global__ void triangleInBox(unsigned int *surface, int *triangle, unsigned int color, float *zbuffer, int xStart, int yStart, int xEnd, int yEnd, int height, float cross2)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x + xStart;
    int idy = threadIdx.y + blockDim.y * blockIdx.y + yStart;
    if (idx < xEnd && idy < yEnd) {
        float cross0 = (triangle[3] - triangle[0]) * (triangle[1] - idy) - (triangle[4] - triangle[1]) * (triangle[0] - idx);
        float cross1 = -1.0 * ((triangle[6] - triangle[0]) * (triangle[1] - idy) - (triangle[7] - triangle[1]) * (triangle[0] - idx));
        int id = idx * height + idy;    
        float u0 = 1.0 - (cross0 + cross1) / cross2;
        float u1 = cross1 / cross2;
        float u2 = cross0 / cross2;
        if(u0 >= 0.0 && u1 >= 0.0 && u2 >= 0.0) 
        {
            float z = triangle[2] * u0 + triangle[5] * u1 + triangle[8] * u2;
            if(z < zbuffer[id])
            {
                zbuffer[id] = z;
                surface[id] = color;   
            }
        }
    }
}
""")
drawTriangleInBox = mod.get_function("triangleInBox")


def drawTriangleGPU(screenSize, surface, triangle, color, zbuffer):
    crossZ = np.float32((triangle[2][0] - triangle[0][0]) * (triangle[1][1] - triangle[0][1])
                        - (triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1]))
    if crossZ != 0:
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
            boxMax[0] = screenSize[0] - 1
        if boxMax[0] < 0:
            return
        boxMax[1] = max(triangle[0][1], triangle[1][1], triangle[2][1])
        if boxMax[1] >= screenSize[1]:
            boxMax[1] = screenSize[1] - 1
        if boxMax[1] < 0:
            return
        size = boxMax - boxMin
        gdim = (int(size[0] // bdim[0] + (size[0] % bdim[0] > 0)),
                int(size[1] // bdim[1] + (size[1] % bdim[1] > 1)))
        drawTriangleInBox(cuda.InOut(surface), cuda.In(triangle), np.uint32(color), cuda.InOut(zbuffer),
                          np.int32(boxMin[0]), np.int32(boxMin[1]), np.int32(boxMax[0]), np.int32(boxMax[1]),
                          np.int32(screenSize[1]), crossZ, block=bdim, grid=gdim)


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


# @njit(void(uint64[:], uint64[:, :], int64[:, :], float64[:, :]))
# def drawPoly(screenSize, surface, points, zbuffer):
#     if points.shape[0] < 3:
#         return
#     triangle = np.empty((3, 3), dtype=np.int64)
#     triangle[0] = points[0]
#     color = uint64(random() * 1000000)
#     for i in range(2, points.shape[0]):
#         triangle[1] = points[i - 1]
#         triangle[2] = points[i]
#         drawTriangle(screenSize, surface, triangle, color, zbuffer)


@njit(void(uint64[:], uint32[:, :], float32[:, :], int64[:, :], float32[:, :], float64))
def drawPolys(screenSize, surface, points, faces, zbuffer, depth):
    for face in range(faces.shape[0]):
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
            if triangle[0][2] > 0 and triangle[1][2] > 0 and triangle[2][2] > 0 and triangle[0][2] <= depth \
                    and triangle[1][2] <= depth and triangle[2][2] <= depth:
                drawTriangle(screenSize, surface, triangle, color, zbuffer)


def drawPolysGPU(screenSize, surface, points, faces, zbuffer, depth):
    for face in range(faces.shape[0]):
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
            if triangle[0][2] > 0 and triangle[1][2] > 0 and triangle[2][2] > 0 and triangle[0][2] <= depth \
                    and triangle[1][2] <= depth and triangle[2][2] <= depth:
                drawTriangleGPU(screenSize, surface, triangle, color, zbuffer)

# @njit(void(uint64[:], uint64[:, :], float64[:, :], int64, int64, float64[:, :]), parallel=True)
# def drawTriangles(screenSize, surface, triangles, n, m, zbuffer):
#     for i in range(n-1):
#         for j in range(m-1):
#             if triangles[(i + 1) * m + j][3] > 0.01 and triangles[i * m + j + 1][3] > 0.01:
#                 triangle = np.empty((3, 3), dtype=np.int64)
#                 for k in prange(3):
#                     triangle[1][k] = round(triangles[(i + 1) * m + j][k] / triangles[(i + 1) * m + j][3])
#                     triangle[2][k] = round(triangles[i * m + j + 1][k] / triangles[i * m + j + 1][3])
#                 if triangles[i * m + j][3] > 0.01:
#                     for k in range(3):
#                         triangle[0][k] = round(triangles[i * m + j][k] / triangles[i * m + j][3])
#                     drawTriangle(screenSize, surface, triangle, uint32(random() * 1000000), zbuffer)
#                 if triangles[(i + 1) * m + j + 1][3] > 0.01:
#                     for k in range(3):
#                         triangle[0][k] = round(triangles[(i + 1) * m + j + 1][k] / triangles[(i + 1) * m + j + 1][3])
#                     drawTriangle(screenSize, surface, triangle, uint32(random() * 1000000), zbuffer)


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

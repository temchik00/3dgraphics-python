import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from random import random

bdim = (8, 8, 1)
gdim_fullscreen = (0, 0, 0)


def initGdim(screenSize):
    global gdim_fullscreen
    gdim_fullscreen = (int(screenSize[0] // bdim[0] + (screenSize[0] % bdim[0] > 0)),
                       int(screenSize[1] // bdim[1] + (screenSize[1] % bdim[1] > 0)))


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


__global__ void clearScreenGPUcore(unsigned int *surface, unsigned int colorClear, int sizeX, int sizeY)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < sizeX && idy < sizeY) {
        surface[idx * sizeY + idy] = colorClear;
    }
}


__global__ void clearBufferGPUcore(float *zBuffer, int clipDist, int sizeX, int sizeY)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < sizeX && idy < sizeY) {
        zBuffer[idx * sizeY + idy] = clipDist;
    }
}
""")
drawTriangleInBox = mod.get_function("triangleInBox")
clearScreenGPUcore = mod.get_function("clearScreenGPUcore")
clearBufferGPUcore = mod.get_function("clearBufferGPUcore")


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
        if size[0] == 0 or size[1] == 0:
            return
        gdim = (int(size[0] // bdim[0] + (size[0] % bdim[0] > 0)),
                int(size[1] // bdim[1] + (size[1] % bdim[1] > 0)))
        drawTriangleInBox(surface, cuda.In(triangle), np.uint32(color), zbuffer,
                          np.int32(boxMin[0]), np.int32(boxMin[1]), np.int32(boxMax[0]), np.int32(boxMax[1]),
                          np.int32(screenSize[1]), crossZ, block=bdim, grid=gdim)


def drawPolysGPU(screenSize, surface, points, faces, zbuffer, depth):
    for face in range(faces.shape[0]):
        if 0 < points[faces[face][0]][2] <= depth:
            color = np.uint32(random() * 1000000)
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
                    drawTriangleGPU(screenSize, surface, triangle, color, zbuffer)


def clearScreenGPU(screen, colorHex, screenSize):
    clearScreenGPUcore(screen, np.uint32(colorHex), np.int32(screenSize[0]),
                       np.int32(screenSize[1]), block=bdim, grid=gdim_fullscreen)


def clearBufferGPU(zBuffer, clipDist, screenSize):
    clearBufferGPUcore(zBuffer, np.int32(clipDist), np.int32(screenSize[0]),
                       np.int32(screenSize[1]), block=bdim, grid=gdim_fullscreen)

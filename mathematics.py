from numba import jit, uint32, void, int32, njit, prange, float64
import numpy as np


@njit(float64[:, :](uint32[:], float64[:], float64[:, :]))
def flatten(screenSize, cameraPos, objectDots):
    matrix = np.eye(4, dtype=np.float64)
    matrix[0][0] = screenSize[0]/2
    matrix[1][1] = screenSize[1]/2
    matrix[0][3] = cameraPos[0] + screenSize[0]/2
    matrix[1][3] = cameraPos[1] + screenSize[1]/2
    matrix[2][3] = 255
    matrix[3][2] = 1.0/cameraPos[2]
    res = np.dot(matrix, objectDots)
    return res.T



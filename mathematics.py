from numba import jit, uint32, void, int32, njit, prange, float64, int64
import numpy as np


@njit(float64[:, :](uint32[:], float64[:], float64[:, :]))
def flatten(screenSize, cameraPos, objectDots):
    matrix = np.eye(4, dtype=np.float64)

    #scale
    matrix[0][0] = screenSize[0]/4
    matrix[1][1] = screenSize[1]/4
    matrix[2][2] = 100
    res = np.dot(matrix, objectDots)

    #shift
    matrix[0][0] = 1
    matrix[1][1] = 1
    matrix[2][2] = 1
    matrix[0][3] = cameraPos[0]
    matrix[1][3] = cameraPos[1]
    matrix[2][3] = cameraPos[2]
    res = np.dot(matrix, res)

    return res.T



from numba import uint64, njit, float64
import numpy as np


@njit(float64[:, :](uint64[:], float64[:], float64[:, :], float64))
def flatten(screenSize, cameraPos, objectDots, focus):
    matrix = np.eye(4, dtype=np.float64)

    # change position
    matrix[0][3] = cameraPos[0]
    matrix[1][3] = cameraPos[1]
    matrix[2][3] = cameraPos[2]
    res = np.dot(matrix, objectDots)

    # projection
    matrix[0][3] = 0
    matrix[1][3] = 0
    matrix[2][3] = 0
    matrix[3][2] = -1 / focus
    res = np.dot(matrix, res)

    # scale
    matrix[3][2] = 0
    matrix[0][0] = screenSize[0] / 2
    matrix[1][1] = screenSize[1] / 2
    matrix[2][2] = 32
    res = np.dot(matrix, res)

    return res.T



from numba import jit, uint32, void, int32, njit, prange, float64
import numpy as np


#@njit(float32[:, :](int32[:], float64[:, :]), parallel=True)
def flatten(cameraPos, objectDots):
    matrix = np.eye(4, dtype=np.float64)
    matrix[3][2] = -1.0/cameraPos[2]
    res = np.dot(matrix, objectDots)
    return res.T



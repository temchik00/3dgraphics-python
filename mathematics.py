from numba import jit, uint32, void, int32, njit, prange, float64
import numpy as np


@njit(int32[:, :](int32[:], float64[:, :]), parallel=True)
def flatten(cameraPos, objectDots):
    matrix = np.eye(4, dtype=np.float64)
    matrix[3][2] = 1.0/cameraPos[2]
    res = np.dot(matrix, objectDots)
    ans = np.empty(res.shape, dtype=np.int32)
    for i in prange(ans.shape[0]):
        for j in prange(3):
            ans[i][j] = round(res[i][j] / res[i][3])
    return ans



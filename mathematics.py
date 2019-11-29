from numba import jit, float64, int64, prange
import numpy as np


scaleMatrix = np.eye(4, dtype=np.float64)
projectMatrix = np.eye(4, dtype=np.float64)
shift = np.eye(4, dtype=np.float64)


def initShiftMatrix(screenSize):
    shift[0][3] = screenSize[0] / 2
    shift[1][3] = screenSize[1] / 2


def initProjectMatrix(screenSize, nearClipPlane, farClipPlane, fov):
    aspect = screenSize[0] / screenSize[1]
    fov = np.radians(fov)
    a = farClipPlane / (farClipPlane - nearClipPlane)
    b = nearClipPlane * farClipPlane / (nearClipPlane - farClipPlane)
    projectMatrix[0][0] = 1/(np.tan(fov/2)*aspect)
    projectMatrix[1][1] = 1/np.tan(fov/2)
    projectMatrix[2][2] = a
    projectMatrix[3][3] = 0
    projectMatrix[2][3] = b
    projectMatrix[3][2] = 1


def initScaleMatrix(screenSize, depth):
    scaleMatrix[0][0] = screenSize[0] / 2
    scaleMatrix[1][1] = screenSize[1] / 2
    scaleMatrix[2][2] = depth - 1


def transform(cameraPos, objectDots, angle):
    # change position
    matrix = np.eye(4, dtype=np.float64)
    matrix[0][3] = -1 * cameraPos[0]
    matrix[1][3] = -1 * cameraPos[1]
    matrix[2][3] = -1 * cameraPos[2]
    res = np.dot(matrix, objectDots)

    ## rotation ##
    # up/down
    rotateX = np.eye(4, dtype=np.float64)
    rotateX[1][1] = np.cos(angle[1])
    rotateX[2][2] = np.cos(angle[1])
    rotateX[1][2] = -1 * np.sin(angle[1])
    rotateX[2][1] = np.sin(angle[1])
    res = np.dot(rotateX, res)

    # left/right
    rotateY = np.eye(4, dtype=np.float64)
    rotateY[0][0] = np.cos(angle[0])
    rotateY[2][2] = np.cos(angle[0])
    rotateY[0][2] = -1 * np.sin(angle[0])
    rotateY[2][0] = np.sin(angle[0])
    res = np.dot(rotateY, res)


    # projection
    res = np.dot(projectMatrix, res)

    # scale
    res = np.dot(scaleMatrix, res)

    # Change camera center
    res = shiftImage(res)

    return res.T


# Calculates image points and moves them, so the camera center is window center
@jit(parallel=True)
def shiftImage(objectDots):
    for point in prange(objectDots.shape[1]):
        for coord in range(3):
            if objectDots[3][point] > 0:
                objectDots[coord][point] = round(objectDots[coord][point] / objectDots[3][point])
            else:
                objectDots[coord][point] = 0
        objectDots[3][point] = 1
    return np.dot(shift, objectDots)

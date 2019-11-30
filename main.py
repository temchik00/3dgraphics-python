import numpy as np
import pygame
from graphics import *
#from landscape import *
from mathematics import initScaleMatrix, transform, initProjectMatrix, initShiftMatrix


play = True
n, m = 2, 2
depth = 256
ang = np.full(2, 0, dtype=np.float64)
cameraPos = np.array([0.0, 0.0, -4.0], dtype=np.float64)
screenSize = np.array((1920, 1080), dtype=np.uint64)
surfArray = np.full(screenSize, 16777215, dtype=np.uint64)
zBuffer = np.full(screenSize, 16777215,  dtype=np.float64)
rotateX = np.eye(3, dtype=np.float64)
rotateY = np.eye(3, dtype=np.float64)
shift = np.empty(3)


pygame.init()
initScaleMatrix(screenSize, depth)
initProjectMatrix(screenSize, 0.1, 10, 80)
initShiftMatrix(screenSize)
screen = pygame.display.set_mode(screenSize, pygame.DOUBLEBUF | pygame.FULLSCREEN)
screen.unlock()
clock = pygame.time.Clock()

# test object's points(pyramid)
points = np.empty((4, 4), dtype=np.float64, order='C')
points[0][0] = 1
points[1][0] = 0
points[2][0] = 2
points[3][0] = 1

points[0][1] = 1
points[1][1] = 0.5
points[2][1] = 1
points[3][1] = 1

points[0][2] = 0
points[1][2] = 1
points[2][2] = 2
points[3][2] = 1

points[0][3] = 2
points[1][3] = 1
points[2][3] = 2
points[3][3] = 1

# faces of test object
faces = np.empty((4, 3), dtype=np.int64, order='C')
faces[0][0] = 0
faces[0][1] = 2
faces[0][2] = 3

faces[1][0] = 0
faces[1][1] = 1
faces[1][2] = 2

faces[2][0] = 0
faces[2][1] = 1
faces[2][2] = 3

faces[3][0] = 1
faces[3][1] = 2
faces[3][2] = 3


while play:
    clearScreen(surfArray, 16777215)
    clearBuffer(zBuffer, depth)
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            play = False
            break
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                play = False
                break

    if not play:
        break

    keys = pygame.key.get_pressed()
    # Changing camera angles
    if keys[pygame.K_LEFT]:
        ang[0] -= 0.05
    elif keys[pygame.K_RIGHT]:
        ang[0] += 0.05
    if keys[pygame.K_UP] and ang[1] > -1.39:
        ang[1] -= 0.05
    elif keys[pygame.K_DOWN] and ang[1] < 1.39:
        ang[1] += 0.05

    # Changing camera position
    if keys[pygame.K_q]:
        shift[1] = -0.2
    elif keys[pygame.K_e]:
        shift[1] = 0.2
    else:
        shift[1] = 0
    if keys[pygame.K_a]:
        shift[0] = -0.2
    elif keys[pygame.K_d]:
        shift[0] = 0.2
    else:
        shift[0] = 0
    if keys[pygame.K_w]:
        shift[2] = 0.3
    elif keys[pygame.K_s]:
        shift[2] = -0.3
    else:
        shift[2] = 0

    # Rotating shift
    rotateX[1][1] = np.cos(ang[1])
    rotateX[2][2] = np.cos(ang[1])
    rotateX[1][2] = np.sin(ang[1])
    rotateX[2][1] = -1 * np.sin(ang[1])
    shift = np.dot(rotateX, shift)

    rotateY[0][0] = np.cos(ang[0])
    rotateY[2][2] = np.cos(ang[0])
    rotateY[0][2] = np.sin(ang[0])
    rotateY[2][0] = -1 * np.sin(ang[0])
    shift = np.dot(rotateY, shift)

    # Applying shift
    cameraPos = np.add(cameraPos, shift)

    # Get points to draw
    triangleMap = transform(cameraPos, points, ang)

    # Draw everything
    drawPolys(screenSize, surfArray, triangleMap, faces, zBuffer, depth)

    pygame.surfarray.blit_array(screen, surfArray)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()

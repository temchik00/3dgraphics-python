import numpy as np
import pygame
from graphics import *
from landscape import *


play = True
n, m = 2, 2

cameraPos = np.array([0.0, 0.0, -20.0], dtype=np.float64)
screenSize = np.array((1920, 1080), dtype=np.uint64)
surfArray = np.full(screenSize, 16777215, dtype=np.uint64)
zBuffer = np.full(screenSize,16777215,  dtype=np.float64)

pygame.init()
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
    clearBuffer(zBuffer, 255)
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

    # moving camera
    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        cameraPos[1] += 0.2
    elif keys[pygame.K_e]:
        cameraPos[1] -= 0.2
    if keys[pygame.K_a]:
        cameraPos[0] += 0.2
    elif keys[pygame.K_d]:
        cameraPos[0] -= 0.2
    if keys[pygame.K_w]:
        cameraPos[2] += 0.3
    elif keys[pygame.K_s]:
        cameraPos[2] -= 0.3

    triangleMap = flatten(screenSize, cameraPos, points, 100.0)
    drawPolys(screenSize, surfArray, triangleMap, faces, zBuffer)
    pygame.surfarray.blit_array(screen, surfArray)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()

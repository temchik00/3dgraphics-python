import numpy as np
import pygame
from graphics import *
from landscape import *


drawSize = 600
play = True
n, m = 2, 2
pos = np.array([0.0, 0.0, -200.0], dtype=np.float64)
screenSize = np.array((1920, 1080), dtype=np.uint32)

pygame.init()
screen = pygame.display.set_mode(screenSize, pygame.DOUBLEBUF)
screen.unlock()
surfArray = np.full(screenSize, 16777215, dtype=np.uint32)
zBuffer = np.empty(screenSize, dtype=np.int32)
clock = pygame.time.Clock()
t = generateMap(10, n, m)
tmp = to2D(t)

tmp = np.empty((4, 4), dtype=np.float, order='C')
tmp[0][0] = 1
tmp[1][0] = 1
tmp[2][0] = 1
tmp[3][0] = 1

tmp[0][1] = 3
tmp[1][1] = 1
tmp[2][1] = 1
tmp[3][1] = 1

tmp[0][2] = 1
tmp[1][2] = 4
tmp[2][2] = 2
tmp[3][2] = 1

tmp[0][3] = 3
tmp[1][3] = 4
tmp[2][3] = 2
tmp[3][3] = 1

points = np.empty((4, 4), dtype=np.float64)
points[0][0] = 1
points[0][1] = 0
points[0][2] = 0
points[0][3] = 1

points[1][0] = 1
points[1][1] = 0.5
points[1][2] = -1
points[1][3] = 1

points[2][0] = 0
points[2][1] = 1
points[2][2] = 0
points[2][3] = 1

points[3][0] = 2
points[3][1] = 1
points[3][2] = 0
points[3][3] = 1

faces = np.empty((4, 3), dtype=np.int32)
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
    if keys[pygame.K_q]:
        pos[1] -= 2
    elif keys[pygame.K_e]:
        pos[1] += 2
    if keys[pygame.K_a]:
        pos[0] += 2
    elif keys[pygame.K_d]:
        pos[0] -= 2
    if keys[pygame.K_w]:
        pos[2] += 0.3
    elif keys[pygame.K_s]:
        pos[2] -= 0.3

    triangleMap = flatten(screenSize, pos, points)
    #drawPoly(screenSize, surfArray, points)
    #drawTriangles(screenSize, surfArray, triangleMap, n, m)
    drawPolys(screenSize, surfArray, triangleMap, faces)
    pygame.surfarray.blit_array(screen, surfArray)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
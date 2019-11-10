import numpy as np
import pygame
from graphics import *
from landscape import *


drawSize = 600
play = True
n, m = 2, 2
pos = [0.0, 0.0, -40.0]
screenSize = np.array((1920, 1080), dtype=np.uint32)

pygame.init()
screen = pygame.display.set_mode(screenSize, pygame.DOUBLEBUF)
screen.unlock()
surfArray = np.full(screenSize, 16777215, dtype=np.uint32)
zBuffer = np.empty(screenSize, dtype=np.int32)
clock = pygame.time.Clock()
t = generateMap(10, n, m)
tmp = to2D(t)

clearScreen(surfArray, 16777215)
clearBuffer(zBuffer, 50)
triangleMap = flatten(np.array((0, 0, -2), order='C'), tmp)
drawTriangles(screenSize, surfArray, triangleMap, n, m)
pygame.surfarray.blit_array(screen, surfArray)
pygame.display.flip()
while play:
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
        pos[1] -= 0.5
    elif keys[pygame.K_e]:
        pos[1] += 0.5
    elif keys[pygame.K_a]:
        pos[0] -= 0.5
    elif keys[pygame.K_d]:
        pos[0] += 0.5
    elif keys[pygame.K_w]:
        pos[2] += 0.5
    elif keys[pygame.K_s]:
        pos[2] -= 0.5

    # triangleMap = flatten(np.array((0, 0, -15), order='C'), tmp)
    # drawTriangles(screenSize, surfArray, triangleMap, 30, 100)
    # pygame.surfarray.blit_array(screen, surfArray)
    # pygame.display.flip()
    clock.tick(60)
pygame.quit()
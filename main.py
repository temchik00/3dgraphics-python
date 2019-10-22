import numpy as np
import pygame
from graphics import *

drawSize = 600
screenSize = np.array((1920, 1080), dtype=np.uint32)
center = (screenSize[0] // 2, screenSize[1] // 2)
pygame.init()
screen = pygame.display.set_mode(screenSize, pygame.DOUBLEBUF)
screen.unlock()
play = True
clock = pygame.time.Clock()
surfArray = np.full(screenSize, 16777215, dtype=np.uint32)

pos = [0.0, 0.0, -40.0]
trianglePos = 0

while play:
    clearScreen(surfArray, 16777215)
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            play = False
            pygame.quit()
            break
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                play = False
                pygame.quit()
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

    drawTriangle(screenSize, surfArray, np.array([[trianglePos, trianglePos], [10+trianglePos, trianglePos],
                                                  [trianglePos, 10+trianglePos]], dtype=np.int32))
    pygame.surfarray.blit_array(screen, surfArray)
    pygame.display.flip()
    if trianglePos < 900:
        trianglePos += 1
    clock.tick(60)
pygame.quit()
import numpy as np
import pygame
from numba import njit, float64
from graphicsGPU import *
from mathematics import initScaleMatrix, transform,\
    initProjectMatrix, initShiftMatrix
from time import time
from objReader import showMenu, readFromFile

file = showMenu()
if file is not None:
    # Preparing variables
    frameCounter = 0
    points, faces = readFromFile(file)
    play = True
    n, m = 2, 2
    depth = 256
    ang = np.full(2, 0, dtype=np.float64)
    cameraPos = np.array([0.0, 0.0, -15.0], dtype=np.float64)
    screenSize = np.array((1920, 1080), dtype=np.uint64)
    surfArray = np.full(screenSize, 16777215, dtype=np.uint32)
    zBuffer = np.full(screenSize, depth,  dtype=np.float32)
    shift = np.empty(3, dtype=np.float64)

    # Init things
    pygame.init()
    initScaleMatrix(screenSize, depth)
    initProjectMatrix(screenSize, 0.5, 40, 80)
    initShiftMatrix(screenSize)
    initGdim(screenSize)
    screen = pygame.display.set_mode(screenSize,
                                     pygame.DOUBLEBUF | pygame.FULLSCREEN)
    screen.unlock()
    clock = pygame.time.Clock()

    # Function for moving camera
    @njit(float64[:](float64[:], float64[:], float64[:]))
    def applyMovement(camPos, camAng, Shift):
        rotateX = np.eye(3, dtype=np.float64)
        rotateY = np.eye(3, dtype=np.float64)

        rotateX[1][1] = np.cos(camAng[1])
        rotateX[2][2] = np.cos(camAng[1])
        rotateX[1][2] = np.sin(camAng[1])
        rotateX[2][1] = -1 * np.sin(camAng[1])
        Shift = np.dot(rotateX, Shift)

        rotateY[0][0] = np.cos(camAng[0])
        rotateY[2][2] = np.cos(camAng[0])
        rotateY[0][2] = np.sin(camAng[0])
        rotateY[2][0] = -1 * np.sin(camAng[0])
        Shift = np.dot(rotateY, Shift)

        # Applying shift
        return np.add(camPos, Shift)

    # Transferring data into device
    surface_gpu = cuda.mem_alloc(surfArray.nbytes)
    cuda.memcpy_htod(surface_gpu, surfArray)
    zBuffer_gpu = cuda.mem_alloc(zBuffer.nbytes)
    cuda.memcpy_htod(zBuffer_gpu, zBuffer)

    # Timing
    timePassed = time()

    # Render's main loop
    while play:
        # Clearing things
        clearScreenGPU(surface_gpu, 16777215, screenSize)
        clearBufferGPU(zBuffer_gpu, depth, screenSize)

        # Implementing application closure
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

        # Handling keyboard input
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

        # Applying shift
        cameraPos = applyMovement(cameraPos, ang, shift)

        # Get points to draw
        triangleMap = transform(cameraPos, points, ang)
        # Draw everything
        drawPolysGPU(screenSize, surface_gpu, triangleMap,
                     faces, zBuffer_gpu, depth)

        # Copy data from device
        cuda.memcpy_dtoh(surfArray, surface_gpu)
        pygame.surfarray.blit_array(screen, surfArray)
        pygame.display.flip()
        frameCounter += 1
        clock.tick(60)

    # Measuring average fps
    timePassed = time() - timePassed
    print(frameCounter / timePassed)
    pygame.quit()

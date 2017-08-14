import random
from math import sqrt

import pygame
import sys
import time
from pygame.locals import *


def drawLine(x1, y1, x2, y2):
    pass


def drawPoint(color, x, y):
    pygame.draw.line(DISPLAYSURF, color, (x, y), (x, y))


def getRandomColor(color):
    if color == 1:
        return BLACK
    elif color == 2:
        return WHITE
    elif color == 3:
        return RED
    elif color == 4:
        return GREEN
    elif color == 5:
        return BLUE

def calculaDistancia(x1, x2, y1, y2):
    return sqrt((x2 - x1) ** 2) + ((y2 - y1) ** 2)


def getAccessPointPosition():
    return [1000, 450]

def twoRay(Pt, ):
    return


WIDTH = 2000
HEIGHT = 900

pygame.init()

# set up the window
DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Drawing')

# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

for x in range(2000):
    for y in range(900):
        color = getRandomColor(random.randint(1, 5))
        drawPoint(color, x, y)
        #drawPoint(BLUE, x, y)

ap = getAccessPointPosition()
drawPoint(RED, ap[0], ap[1])

# run the game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    time.sleep(5)
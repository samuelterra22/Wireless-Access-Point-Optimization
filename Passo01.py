import random
import sys
import time
from math import sqrt, pi

import pygame
from pygame.locals import *

WIDTH = 2000
HEIGHT = 900
CHANNEL = 9

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def draw_line(x1, y1, x2, y2):
    pass


def draw_point(color, x, y):
    pygame.draw.line(DISPLAYSURF, color, (x, y), (x, y))


def get_random_color(color):
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


def calc_distance(x1, x2, y1, y2):
    return sqrt((x2 - x1) ** 2) + ((y2 - y1) ** 2)


def get_access_point_position():
    return [1000, 450]


def frequency():
    return (2.407 + (5 * CHANNEL) / 1000) * 10 ** 9


def wave_length():
    C = 299792458
    return C/frequency()


def free_space_model(Pt, Gt, Gr, lamb, d, L):
    return (Pt*Gt*Gr*(pow(lamb, 2))) / pow((4*pi), 2) * pow(d, 2) * L


pygame.init()

# set up the window
DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Drawing')

# set up the colors


for x in range(2000):
    for y in range(900):
        color = get_random_color(random.randint(1, 5))
        draw_point(color, x, y)
        # draw_point(BLUE, x, y)

ap = get_access_point_position()
draw_point(RED, ap[0], ap[1])

# run the game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    time.sleep(5)

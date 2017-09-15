import sys

#   https://stackoverflow.com/questions/20034023/maximum-recursion-depth-exceeded-in-comparison
#
#
#
import numpy as np

sys.setrecursionlimit(10000 * 10000)


def solveMaze(Maze, position, point):
    N = len(Maze)  # lin
    M = len(Maze[0])  # col

    # print(position)

    # returns a list of the paths taken
    if position == point:
        return [point]

    x, y = position

    # if x <= point[0] and y <= point[1]:
    #
    #     if y + 1 < M and x + 1 < N and x <= point[0] and y <= point[1] and Maze[x + 1][y + 1] == 0:
    #         a = solveMaze(Maze, (x + 1, y + 1), point)
    #         if a is not None:
    #             return [(x, y)] + a
    #
    #     if x + 1 < N and Maze[x + 1][y] == 0:
    #         b = solveMaze(Maze, (x + 1, y), point)
    #         if b is not None:
    #             return [(x, y)] + b
    #
    #     if y + 1 < M and Maze[x][y + 1] == 0:
    #         c = solveMaze(Maze, (x, y + 1), point)
    #         if c is not None:
    #             return [(x, y)] + c
    # # ---------------------------------------------------------
    # elif x >= point[0] and y >= point[1]:
    #
    #     if y - 1 >= 0 and x - 1 >= 0 and x >= point[0] and y >= point[1] and Maze[x - 1][y - 1] == 0:
    #         d = solveMaze(Maze, (x - 1, y - 1), point)
    #         if d is not None:
    #             return [(x, y)] + d
    #
    #     if x - 1 >= 0 and Maze[x - 1][y] == 0:
    #         e = solveMaze(Maze, (x - 1, y), point)
    #         if e is not None:
    #             return [(x, y)] + e
    #
    #     if y - 1 >= 0 and Maze[x][y - 1] == 0:
    #         f = solveMaze(Maze, (x, y - 1), point)
    #         if f is not None:
    #             return [(x, y)] + f

    if y + 1 < M and x + 1 < N and Maze[x + 1][y + 1] == 0:
        a = solveMaze(Maze, (x + 1, y + 1), point)
        if a is not None:
            return [(x, y)] + a

    if x + 1 < N and Maze[x + 1][y] == 0:
        b = solveMaze(Maze, (x + 1, y), point)
        if b is not None:
            return [(x, y)] + b

    if y + 1 < M and Maze[x][y + 1] == 0:
        c = solveMaze(Maze, (x, y + 1), point)
        if c is not None:
            return [(x, y)] + c


Maze = np.zeros(shape=(10, 10))

import pygame
import sys
import time
from pygame.locals import *

pygame.init()
s = 50
DISPLAYSURF = pygame.display.set_mode((len(Maze)*s, len(Maze[0])*s), 0, 32)
pygame.display.set_caption('Drawing')

# set up the colors
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

solucao = solveMaze(Maze, (0, 0), (5, 9))
x=0
y=1

for i in range(len(Maze)):
    pygame.draw.line(DISPLAYSURF, GRAY, (s*i,s*0), (s*i,s*len(Maze[0])) )

for j in range(len(Maze[0])):
    pygame.draw.line(DISPLAYSURF, GRAY, (s*0,s*j), (s*len(Maze),s*j) )

for i in range(1, len(solucao)-1):
    pygame.draw.line(DISPLAYSURF, GREEN, (s*solucao[i-1][x], s*solucao[i-1][y]), (s*solucao[i][x], s*solucao[i][y]))


while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
    #sleep ou yield
    time.sleep(5)
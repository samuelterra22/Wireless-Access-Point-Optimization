#from PIL import Image, ImageDraw

#im = Image.new('RGBA', (1000, 500), (0, 255, 0, 0))
#draw = ImageDraw.Draw(im)
# draw.line(((0, 0), (100, 100)), fill=64)

#im.show()


import pygame
import sys
import time
from pygame.locals import *

pygame.init()

# set up the window
DISPLAYSURF = pygame.display.set_mode((2000, 900), 0, 32)
pygame.display.set_caption('Drawing')

# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# draw on the surface object
# pygame.draw.line(DISPLAYSURF, BLUE, (0, 0), (50, 50))
# pygame.draw.line(DISPLAYSURF, BLUE, (50, 50), (30, 20))

pygame.draw.line(DISPLAYSURF, BLUE, (1015, 462), (1015, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (920, 462), (920, 670))
pygame.draw.line(DISPLAYSURF, BLUE, (569, 374), (569, 0))
pygame.draw.line(DISPLAYSURF, BLUE, (821, 374), (821, 0))
pygame.draw.line(DISPLAYSURF, BLUE, (826, 462), (826, 670))
pygame.draw.line(DISPLAYSURF, BLUE, (0, 374), (569, 375))
pygame.draw.line(DISPLAYSURF, BLUE, (192, 462), (192, 670))
pygame.draw.line(DISPLAYSURF, BLUE, (381, 462), (381, 670))
pygame.draw.line(DISPLAYSURF, BLUE, (821, 0), (569, 0))
pygame.draw.line(DISPLAYSURF, BLUE, (1202, 167), (1202, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (1013, 374), (1013, 167))
pygame.draw.line(DISPLAYSURF, BLUE, (821, 167), (1202, 167))
pygame.draw.line(DISPLAYSURF, BLUE, (1202, 374), (821, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (381, 374), (381, 167))
pygame.draw.line(DISPLAYSURF, BLUE, (192, 167), (192, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (508, 462), (508, 670))
pygame.draw.line(DISPLAYSURF, BLUE, (622, 669), (636, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (636, 462), (636, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (0, 462), (1202, 462))
pygame.draw.line(DISPLAYSURF, BLUE, (0, 670), (1202, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (0, 167), (0, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (569, 167), (0, 167))



pixObj = pygame.PixelArray(DISPLAYSURF)
pixObj[380][280] = BLACK
pixObj[382][282] = BLACK
pixObj[384][284] = BLACK
pixObj[386][286] = BLACK
pixObj[388][288] = BLACK
del pixObj

# run the game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONUP:
            print(pygame.mouse.get_pos())

    pygame.display.update()
    #sleep ou yield
    time.sleep(5)
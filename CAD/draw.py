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

pygame.draw.line(DISPLAYSURF, BLUE, (944, 466), (944, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 469), (1011, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (1015, 469), (1015, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (406, 378), (569, 378))
pygame.draw.line(DISPLAYSURF, BLUE, (406, 374), (566, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (383, 378), (383, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (406, 378), (406, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (920, 469), (920, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (917, 469), (917, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 482), (964, 482))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 481), (964, 481))
pygame.draw.line(DISPLAYSURF, BLUE, (944, 469), (1011, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (944, 466), (1016, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (826, 469), (893, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (826, 481), (872, 481))
pygame.draw.line(DISPLAYSURF, BLUE, (826, 482), (872, 482))
pygame.draw.line(DISPLAYSURF, BLUE, (962, 479), (962, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (963, 479), (963, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (922, 466), (915, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (922, 469), (920, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (893, 466), (893, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (922, 466), (922, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (915, 466), (915, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (917, 469), (915, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (915, 469), (917, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (915, 466), (922, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (920, 469), (922, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (874, 479), (874, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (875, 479), (875, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 669), (920, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (917, 669), (826, 670))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 546), (958, 556))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 576), (920, 576))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 577), (920, 577))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 607), (920, 607))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 608), (920, 608))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 597), (958, 607))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 632), (958, 633))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 632), (958, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 632), (958, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 633), (959, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 633), (959, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 632), (958, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 609), (958, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 608), (959, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 608), (959, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 609), (958, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 609), (958, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 609), (958, 608))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 609), (935, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (935, 609), (935, 610))
pygame.draw.line(DISPLAYSURF, BLUE, (935, 610), (958, 610))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 610), (958, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 608), (959, 608))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 633), (959, 633))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 633), (958, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 633), (959, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 579), (958, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (940, 579), (958, 579))
pygame.draw.line(DISPLAYSURF, BLUE, (940, 578), (940, 579))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 578), (940, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 578), (958, 577))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 578), (958, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 578), (958, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 577), (959, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 577), (959, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 578), (958, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 596), (958, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 597), (959, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 597), (959, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 596), (958, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 596), (958, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 596), (958, 597))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 597), (959, 597))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 597), (959, 608))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 576), (959, 577))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 557), (958, 556))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 557), (958, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 557), (958, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 556), (959, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 556), (959, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 557), (958, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 574), (958, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 576), (959, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 576), (959, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 574), (958, 575))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 575), (958, 575))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 575), (958, 576))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 575), (940, 575))
pygame.draw.line(DISPLAYSURF, BLUE, (940, 575), (940, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (940, 574), (958, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 574), (958, 575))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 556), (959, 556))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 546), (959, 556))
pygame.draw.line(DISPLAYSURF, BLUE, (998, 652), (998, 651))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 652), (998, 652))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 651), (998, 651))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 633), (998, 633))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 632), (998, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (998, 632), (998, 633))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 614), (998, 614))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 613), (998, 613))
pygame.draw.line(DISPLAYSURF, BLUE, (998, 613), (998, 614))
pygame.draw.line(DISPLAYSURF, BLUE, (998, 596), (998, 594))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 596), (998, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (1011, 594), (998, 594))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 514), (920, 514))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 512), (920, 512))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 512), (959, 514))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 546), (920, 546))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 544), (920, 544))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 514), (959, 524))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 524), (959, 524))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 543), (958, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (940, 543), (958, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (940, 543), (940, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 543), (940, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 543), (958, 544))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 543), (958, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 543), (958, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 544), (959, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 544), (959, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 543), (958, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 526), (958, 526))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 524), (959, 526))
pygame.draw.line(DISPLAYSURF, BLUE, (959, 524), (959, 526))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 526), (958, 525))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 525), (958, 525))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 525), (958, 524))
pygame.draw.line(DISPLAYSURF, BLUE, (958, 514), (958, 524))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 514), (879, 524))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 525), (879, 524))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 525), (879, 525))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 526), (879, 525))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 524), (878, 526))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 524), (878, 526))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 526), (879, 526))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 543), (879, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 544), (878, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 544), (878, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 543), (879, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 543), (879, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 543), (879, 544))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 543), (897, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (897, 543), (897, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (897, 543), (879, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 543), (879, 543))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 524), (878, 524))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 514), (878, 524))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 544), (917, 544))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 546), (917, 546))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 512), (878, 514))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 512), (917, 512))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 514), (917, 514))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 546), (878, 556))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 556), (878, 556))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 574), (879, 575))
pygame.draw.line(DISPLAYSURF, BLUE, (897, 574), (879, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (897, 575), (897, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 575), (897, 575))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 575), (879, 576))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 575), (879, 575))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 574), (879, 575))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 576), (878, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 576), (878, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 574), (879, 574))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 557), (879, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 556), (878, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 556), (878, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 557), (879, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 557), (879, 557))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 557), (879, 556))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 576), (878, 577))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 597), (878, 608))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 597), (878, 597))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 596), (879, 597))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 596), (879, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 596), (879, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 597), (878, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 597), (878, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 596), (879, 596))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 578), (879, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 577), (878, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 577), (878, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 578), (879, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 578), (879, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 578), (879, 577))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 578), (897, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (897, 578), (897, 579))
pygame.draw.line(DISPLAYSURF, BLUE, (897, 579), (879, 579))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 579), (879, 578))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 633), (878, 633))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 608), (878, 608))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 610), (879, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (902, 610), (879, 610))
pygame.draw.line(DISPLAYSURF, BLUE, (902, 609), (902, 610))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 609), (902, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 609), (879, 608))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 609), (879, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 609), (879, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 608), (878, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 608), (878, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 609), (879, 609))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 632), (879, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 633), (878, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 633), (878, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 632), (879, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 632), (879, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 632), (879, 633))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 597), (879, 607))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 608), (917, 608))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 607), (917, 607))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 577), (917, 577))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 576), (917, 576))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 546), (879, 556))
pygame.draw.line(DISPLAYSURF, BLUE, (878, 633), (878, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (879, 633), (879, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (566, 171), (566, 0))
pygame.draw.line(DISPLAYSURF, BLUE, (569, 378), (569, 3))
pygame.draw.line(DISPLAYSURF, BLUE, (821, 378), (821, 3))
pygame.draw.line(DISPLAYSURF, BLUE, (825, 171), (825, 0))
pygame.draw.line(DISPLAYSURF, BLUE, (1038, 469), (1198, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (1038, 466), (1038, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (1016, 466), (1016, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (797, 466), (631, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (797, 469), (636, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (893, 466), (820, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (820, 466), (820, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (797, 466), (797, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (508, 469), (511, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (822, 669), (636, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (822, 469), (820, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (1015, 469), (1016, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (822, 469), (822, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (826, 469), (826, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (188, 469), (188, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (3, 466), (163, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (3, 378), (163, 378))
pygame.draw.line(DISPLAYSURF, BLUE, (192, 469), (192, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (377, 469), (377, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (381, 469), (381, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (377, 669), (192, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (374, 466), (383, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (374, 469), (377, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (352, 466), (352, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (406, 466), (406, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (383, 466), (383, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (374, 466), (374, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (381, 469), (383, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (406, 466), (511, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (3, 469), (163, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (406, 469), (504, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (825, 0), (566, 0))
pygame.draw.line(DISPLAYSURF, BLUE, (821, 3), (569, 3))
pygame.draw.line(DISPLAYSURF, BLUE, (1198, 174), (1198, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (1202, 171), (1202, 673))
pygame.draw.line(DISPLAYSURF, BLUE, (1013, 374), (1013, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (1009, 374), (1009, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (825, 171), (1202, 171))
pygame.draw.line(DISPLAYSURF, BLUE, (1009, 175), (825, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (1016, 378), (1007, 378))
pygame.draw.line(DISPLAYSURF, BLUE, (1016, 374), (1013, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (1038, 378), (1038, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (984, 378), (984, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (1007, 378), (1007, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (1016, 378), (1016, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (1009, 374), (1007, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (984, 378), (821, 378))
pygame.draw.line(DISPLAYSURF, BLUE, (984, 374), (825, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (1198, 374), (1038, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (1198, 378), (1038, 378))
pygame.draw.line(DISPLAYSURF, BLUE, (825, 374), (825, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (566, 374), (566, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (3, 374), (163, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (381, 374), (383, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (374, 378), (383, 378))
pygame.draw.line(DISPLAYSURF, BLUE, (374, 378), (374, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (352, 378), (352, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (186, 374), (188, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (381, 374), (381, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (377, 374), (377, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (188, 174), (188, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (192, 174), (192, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (381, 669), (391, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (467, 632), (504, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (467, 632), (467, 649))
pygame.draw.line(DISPLAYSURF, BLUE, (463, 628), (504, 628))
pygame.draw.line(DISPLAYSURF, BLUE, (463, 628), (463, 649))
pygame.draw.line(DISPLAYSURF, BLUE, (504, 632), (504, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (467, 669), (504, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (463, 667), (467, 667))
pygame.draw.line(DISPLAYSURF, BLUE, (463, 649), (467, 649))
pygame.draw.line(DISPLAYSURF, BLUE, (463, 667), (463, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (467, 667), (467, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (381, 669), (463, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (504, 469), (504, 628))
pygame.draw.line(DISPLAYSURF, BLUE, (508, 469), (508, 628))
pygame.draw.line(DISPLAYSURF, BLUE, (504, 632), (504, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (549, 603), (549, 606))
pygame.draw.line(DISPLAYSURF, BLUE, (546, 628), (528, 628))
pygame.draw.line(DISPLAYSURF, BLUE, (546, 599), (546, 606))
pygame.draw.line(DISPLAYSURF, BLUE, (546, 632), (528, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (508, 632), (508, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (508, 669), (546, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (622, 673), (636, 673))
pygame.draw.line(DISPLAYSURF, BLUE, (549, 669), (632, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (511, 628), (511, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (528, 628), (528, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (511, 628), (508, 628))
pygame.draw.line(DISPLAYSURF, BLUE, (511, 632), (508, 632))
pygame.draw.line(DISPLAYSURF, BLUE, (632, 603), (549, 603))
pygame.draw.line(DISPLAYSURF, BLUE, (632, 599), (546, 599))
pygame.draw.line(DISPLAYSURF, BLUE, (546, 632), (546, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (632, 469), (632, 599))
pygame.draw.line(DISPLAYSURF, BLUE, (549, 606), (546, 606))
pygame.draw.line(DISPLAYSURF, BLUE, (549, 626), (546, 626))
pygame.draw.line(DISPLAYSURF, BLUE, (546, 626), (546, 628))
pygame.draw.line(DISPLAYSURF, BLUE, (549, 626), (549, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (636, 469), (636, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (533, 466), (533, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (511, 466), (511, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (533, 466), (636, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (533, 469), (632, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (1198, 469), (1198, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (1038, 466), (1198, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (1015, 669), (1198, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (0, 673), (1202, 673))
pygame.draw.line(DISPLAYSURF, BLUE, (632, 603), (632, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (566, 174), (381, 175))
pygame.draw.line(DISPLAYSURF, BLUE, (3, 469), (3, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (0, 171), (0, 673))
pygame.draw.line(DISPLAYSURF, BLUE, (3, 174), (3, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (192, 469), (352, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (188, 669), (3, 669))
pygame.draw.line(DISPLAYSURF, BLUE, (566, 171), (0, 171))
pygame.draw.line(DISPLAYSURF, BLUE, (377, 174), (366, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (186, 466), (186, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (163, 466), (163, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (163, 378), (163, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (186, 378), (186, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (186, 466), (352, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (186, 469), (188, 469))
pygame.draw.line(DISPLAYSURF, BLUE, (186, 378), (352, 378))
pygame.draw.line(DISPLAYSURF, BLUE, (3, 378), (3, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (1198, 378), (1198, 466))
pygame.draw.line(DISPLAYSURF, BLUE, (1013, 174), (1198, 174))
pygame.draw.line(DISPLAYSURF, BLUE, (188, 175), (3, 175))
pygame.draw.line(DISPLAYSURF, BLUE, (377, 175), (192, 175))
pygame.draw.line(DISPLAYSURF, BLUE, (192, 374), (352, 374))
pygame.draw.line(DISPLAYSURF, BLUE, (374, 374), (377, 374))



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
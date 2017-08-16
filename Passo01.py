import tkinter as tk
from datetime import datetime
from math import sqrt, pi, log10

import numpy as np
import pygame


def get_monitor_size():
    root = tk.Tk()
    return root.winfo_screenwidth(), root.winfo_screenheight()


WIDTH = get_monitor_size()[0] - 100  # Retira 100pxs para folga
HEIGHT = get_monitor_size()[1] - 100  # Retira 100pxs para folga
CHANNEL = 9

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.init()


def draw_line(x1, y1, x2, y2):
    pygame.draw.line(DISPLAYSURF, color, (x1, y1), (x2, y2))


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


def calc_distance(x1, y1, x2, y2):
    return sqrt(pow((x1 - x2), 2.0) + pow((y1 - y2), 2.0))


def get_access_point_position():
    return [1000, 450]


def frequency():
    return (2.407 + (5 * CHANNEL) / 1000) * 10 ** 9


def wave_length():
    """
    Velocidade da luz / frequência do canal
    """
    C = 299792458
    return C / frequency()


def path_loss(d):
    """
    Perda no caminho (Path Loss) mensurado em dB
    :param d: Distâcia
    :return: Perda no caminho
    """
    return 20 * log10((4 * pi * d) / wave_length())


def two_ray_ground_reflection_model(Pt, Gt, Gr, Ht, Hr, d, L):
    """
    Pr
    """
    return (Pt * Gt * Gr * pow(Ht, 2) * pow(Hr, 2)) / (pow(d, 4) * L)


def free_space_model(Pt, Gt, Gr, lamb, d, L):
    """
    Pr
    """
    return (Pt * Gt * Gr * (pow(lamb, 2))) / (pow((4 * pi), 2) * pow(d, 2) * L)


def log_distance(d0, d, gamma):
    """
    Modelo logaritmo de perda baseado em resultados experimentais. Independe da frequência do sinal transmitido
    e do ganho das antenas transmissora e receptora
    """
    # return path_loss(d) + 10 * gamma * log10(d / d0)
    return 17 - (60 + 10 * gamma * log10(d / d0))  # igual está na tabela


def propagation_model():
    return 0


def imprime_matriz_resultados(matriz):
    print("Escrevendo matriz no arquivo de saida...")
    print("Dimanções na matriz: " + str(np.shape(matriz)))
    f = open('saida_passo_01', 'w')
    for linha in matriz:
        for valor in linha:
            f.write(str(valor) + "\t")
        f.write('\n')
    f.close()
    print("Matriz salva no arquivo.")


inicio = datetime.now()

# set up the window
DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Simulando...')

matrix_results = np.zeros(shape=(WIDTH, HEIGHT))

for x in range(WIDTH):
    for y in range(HEIGHT):
        color = BLUE
        draw_point(color, x, y)
        matrix_results[x][y] = propagation_model()

ap = get_access_point_position()
draw_point(RED, ap[0], ap[1])

pygame.display.update()
imprime_matriz_resultados(matrix_results)

fim = datetime.now()

print("\nInicio: \t" + str(inicio.time()))
print("Fim: \t\t" + str(fim.time()))
print("Duração \t" + str((fim - inicio).seconds) + " segundos.\n")

pygame.display.set_caption('Simulação terminada')
input('Precione qualquer tecla para encerrar a aplicação.')

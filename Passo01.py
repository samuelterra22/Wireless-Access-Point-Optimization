import tkinter as tk
from datetime import datetime
from math import sqrt, pi, log10

import numpy as np
import pygame


def get_monitor_size():
    """
    Método que identifica o tamanho da tela do computador.
    :return: Retorna os valores de largura e altura.
    """
    root = tk.Tk()
    return root.winfo_screenwidth(), root.winfo_screenheight()


WIDTH = get_monitor_size()[0] - 100  # Retira 100pxs para folga
HEIGHT = get_monitor_size()[1] - 100  # Retira 100pxs para folga
CHANNEL = 9

COLORS = [
    '#0C0786', '#100787', '#130689', '#15068A', '#18068B', '#1B068C', '#1D068D', '#1F058E', '#21058F', '#230590',
    '#250591', '#270592', '#290593', '#2B0594', '#2D0494', '#2F0495', '#310496', '#330497', '#340498', '#360498',
    '#380499', '#3A049A', '#3B039A', '#3D039B', '#3F039C', '#40039C', '#42039D', '#44039E', '#45039E', '#47029F',
    '#49029F', '#4A02A0', '#4C02A1', '#4E02A1', '#4F02A2', '#5101A2', '#5201A3', '#5401A3', '#5601A3', '#5701A4',
    '#5901A4', '#5A00A5', '#5C00A5', '#5E00A5', '#5F00A6', '#6100A6', '#6200A6', '#6400A7', '#6500A7', '#6700A7',
    '#6800A7', '#6A00A7', '#6C00A8', '#6D00A8', '#6F00A8', '#7000A8', '#7200A8', '#7300A8', '#7500A8', '#7601A8',
    '#7801A8', '#7901A8', '#7B02A8', '#7C02A7', '#7E03A7', '#7F03A7', '#8104A7', '#8204A7', '#8405A6', '#8506A6',
    '#8607A6', '#8807A5', '#8908A5', '#8B09A4', '#8C0AA4', '#8E0CA4', '#8F0DA3', '#900EA3', '#920FA2', '#9310A1',
    '#9511A1', '#9612A0', '#9713A0', '#99149F', '#9A159E', '#9B179E', '#9D189D', '#9E199C', '#9F1A9B', '#A01B9B',
    '#A21C9A', '#A31D99', '#A41E98', '#A51F97', '#A72197', '#A82296', '#A92395', '#AA2494', '#AC2593', '#AD2692',
    '#AE2791', '#AF2890', '#B02A8F', '#B12B8F', '#B22C8E', '#B42D8D', '#B52E8C', '#B62F8B', '#B7308A', '#B83289',
    '#B93388', '#BA3487', '#BB3586', '#BC3685', '#BD3784', '#BE3883', '#BF3982', '#C03B81', '#C13C80', '#C23D80',
    '#C33E7F', '#C43F7E', '#C5407D', '#C6417C', '#C7427B', '#C8447A', '#C94579', '#CA4678', '#CB4777', '#CC4876',
    '#CD4975', '#CE4A75', '#CF4B74', '#D04D73', '#D14E72', '#D14F71', '#D25070', '#D3516F', '#D4526E', '#D5536D',
    '#D6556D', '#D7566C', '#D7576B', '#D8586A', '#D95969', '#DA5A68', '#DB5B67', '#DC5D66', '#DC5E66', '#DD5F65',
    '#DE6064', '#DF6163', '#DF6262', '#E06461', '#E16560', '#E26660', '#E3675F', '#E3685E', '#E46A5D', '#E56B5C',
    '#E56C5B', '#E66D5A', '#E76E5A', '#E87059', '#E87158', '#E97257', '#EA7356', '#EA7455', '#EB7654', '#EC7754',
    '#EC7853', '#ED7952', '#ED7B51', '#EE7C50', '#EF7D4F', '#EF7E4E', '#F0804D', '#F0814D', '#F1824C', '#F2844B',
    '#F2854A', '#F38649', '#F38748', '#F48947', '#F48A47', '#F58B46', '#F58D45', '#F68E44', '#F68F43', '#F69142',
    '#F79241', '#F79341', '#F89540', '#F8963F', '#F8983E', '#F9993D', '#F99A3C', '#FA9C3B', '#FA9D3A', '#FA9F3A',
    '#FAA039', '#FBA238', '#FBA337', '#FBA436', '#FCA635', '#FCA735', '#FCA934', '#FCAA33', '#FCAC32', '#FCAD31',
    '#FDAF31', '#FDB030', '#FDB22F', '#FDB32E', '#FDB52D', '#FDB62D', '#FDB82C', '#FDB92B', '#FDBB2B', '#FDBC2A',
    '#FDBE29', '#FDC029', '#FDC128', '#FDC328', '#FDC427', '#FDC626', '#FCC726', '#FCC926', '#FCCB25', '#FCCC25',
    '#FCCE25', '#FBD024', '#FBD124', '#FBD324', '#FAD524', '#FAD624', '#FAD824', '#F9D924', '#F9DB24', '#F8DD24',
    '#F8DF24', '#F7E024', '#F7E225', '#F6E425', '#F6E525', '#F5E726', '#F5E926', '#F4EA26', '#F3EC26', '#F3EE26',
    '#F2F026', '#F2F126', '#F1F326', '#F0F525', '#F0F623', '#EFF821'
]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Inicia o PyGame
pygame.init()


def hex_to_rgb(hex):
    """
    Método responsável por converter uma cor no formato hexadecial para um RGB.
    :param hex: Valor em hexadecimal da cor.
    :return: Tupla representando a cor em formato RGB.
    """
    hex = str(hex).lstrip('#')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def draw_line(x1, y1, x2, y2):
    pygame.draw.line(DISPLAYSURF, color, (x1, y1), (x2, y2))


def draw_point(color, x, y):
    pygame.draw.line(DISPLAYSURF, color, (x, y), (x, y))


def calc_distance(x1, y1, x2, y2):
    """
    Método responsável por realizar o calculo da distância entre dois pontos no plano cartesiano.
    :param x1: Valor de X no ponto 1.
    :param y1: Valor de Y no ponto 1.
    :param x2: Valor de X no ponto 2.
    :param y2: Valor de Y no ponto 2.
    :return: Retorna um valor float representando a distância dos pontos informados.
    """
    return sqrt(pow((x1 - x2), 2.0) + pow((y1 - y2), 2.0))


def get_access_point_position():
    # return [1000, 450]
    # return [0, 0]
    # return [100, 900]
    return [159, 357]


def frequency():
    """
    Calcula a frequência de acordo com o canal
    :return: Frequência do canal
    """
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
    e do ganho das antenas transmissora e receptora.
    Livro Comunicações em Fio - Pricipios e Práticas - Rappaport (páginas 91-92)
    :param d0: Distância do ponto de referência d0
    :param d:
    :param gamma: Valor da constante de propagação que difere para cada tipo de ambiente.
    :return:
    """
    # return path_loss(d) + 10 * gamma * log10(d / d0)
    return 17 - (60 + 10 * gamma * log10(d / d0))  # igual está na tabela


def propagation_model(x, y):
    access_point = get_access_point_position()
    d = calc_distance(x, y, access_point[0], access_point[1])
    if d == 0:
        d = 1
    gamma = 5

    return log_distance(1, d, gamma)


def print_matriz(matriz):
    """
    Método responsável por imprimir a matriz em um arquivo
    :param matriz: Matriz N x M
    :return: None
    """
    print("Escrevendo matriz no arquivo de saida...")
    print("Dimanções na matriz: " + str(np.shape(matriz)))
    f = open('saida_passo_01', 'w')
    for linha in matriz:
        for valor in linha:
            f.write(str(valor) + "\t")
        f.write('\n')
    f.close()
    print("Matriz salva no arquivo.")


def get_percentage_of_range(min, max, x):
    """
    Método responsável por retornar a porcentagem de acordo com um respectivo intervalo
    :param min: Valor mínimo do intervalo
    :param max: Valor máximo do intervalo
    :param x: Valor que está no intervalo de min-max que deseja saber sua respectiva porcentagem
    :return: Retorna uma porcentagem que está de acordo com o intervalo min-max
    """
    return ((x - min) / (max - min)) * 100


def get_value_in_list(percent, list):
    """
    Método retorna o valor de uma posição de uma lista. A posição é calculada de acordo a porcentagem.
    :param percent: Valor float representando a porcentagem
    :param list: Lista com n números
    :return: Retorna a cor da posição calculada
    """
    position = (percent / 100) * len(list)
    if position < 1:
        position = 1
    elif position >= len(list):
        position = len(list)
    return hex_to_rgb(list[int(position - 1)])


def get_color_of_interval(min, max, x):
    """
    Este método retorna uma cor de acordo com o valor que está entre o intervalo min-max. Em outras palavras,
    este método transforma um número em uma cor dentro de uma faixa informada.
    :param min: Valor mínimo do intervalo
    :param max: Valor máximo do intervalo
    :param x: Valor que está dentro do intervalo e que deseja saber sua cor
    :return: Retorna uma tupla representando um cor no formato RGB.
    """
    percentage = get_percentage_of_range(min, max, x)
    color = get_value_in_list(percentage, COLORS)
    # print('Color: ' + str(color))
    return color


print("Iniciando simulação.")

# Marca o inicio da simulação
inicio = datetime.now()

# Configura o tamanho da janela
DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Simulando...')

# Cria uma matriz para guardar os resultados calculados
matrix_results = np.zeros(shape=(WIDTH, HEIGHT))

print("Posição do access point: " + str(get_access_point_position()))

# Preenche a matriz de resultados usando um modelo de propagação
for x in range(WIDTH):
    for y in range(HEIGHT):
        value = propagation_model(x, y)
        matrix_results[x][y] = value

# Guarda os valores máximo e mínimo da matriz
matrix_max_value = matrix_results.max()
matrix_min_value = matrix_results.min()

# Lê os valores da matriz que contêm valores calculados e colore
for x in range(WIDTH):
    for y in range(HEIGHT):
        color = get_color_of_interval(matrix_min_value, matrix_max_value, matrix_results[x][y])
        draw_point(color, x, y)

# Pinta de vermelho a posição do Access Point
ap = get_access_point_position()
draw_point(RED, ap[0], ap[1])

# Atualiza a janela do PyGame para que exiba a imagem
pygame.display.update()
# Grava os valores da matriz no arquivo
print_matriz(matrix_results)

# Marca o fim da simulação
fim = datetime.now()

print("\nInicio: \t" + str(inicio.time()))
print("Fim: \t\t" + str(fim.time()))
print("Duração: \t" + str((fim - inicio).seconds) + " segundos.\n")

pygame.display.set_caption('Simulação terminada')
print('Simulação terminada.')

print("Maior valor da matriz: " + str(matrix_max_value))
print("Menor valor da matriz: " + str(matrix_min_value))

input('\nPrecione qualquer tecla para encerrar a aplicação.')

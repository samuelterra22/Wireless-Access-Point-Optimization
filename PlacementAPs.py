#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import profile
import random as rd

import ezdxf
import numpy as np
import pygame
import matplotlib.pyplot as plt

from math import sqrt, log10, exp
from random import random

from colour import Color
from numba import cuda, jit

import cProfile

"""
Algoritmo que realiza a simulação da propagação do sinal wireless de determinado ambiente 2D de acordo com um Access
Point Informado.
"""


@jit
def read_walls_from_dxf(dxf_path, escala):
    """
    Método responsável por ler um arquivo DXF e filtrar pela camada ARQ as paredes do ambiente.
    :param escala:
    :param dxf_path: Caminho do arquivo de entrada, sendo ele no formato DFX.
    :return: Retorna uma lista contendo em cada posição, uma lista de quatro elementos, sendo os dois primeiros
    referêntes ao ponto inicial da parede e os dois ultimo referênte ao ponto final da parede.
    """
    dwg = ezdxf.readfile(dxf_path)

    walls = []

    modelspace = dwg.modelspace()

    # TODO 7 por que?
    # escala = 7

    xMin = -1
    yMin = -1
    for e in modelspace:
        if e.dxftype() == 'LINE' and e.dxf.layer == 'ARQ':
            if e.dxf.start[0] < xMin or xMin == -1:
                xMin = e.dxf.start[0]
            if e.dxf.start[1] < yMin or yMin == -1:
                yMin = e.dxf.start[1]

    for e in modelspace:
        if e.dxftype() == 'LINE' and e.dxf.layer == 'ARQ':
            line = [
                int((e.dxf.start[0] - xMin) * escala),
                int((e.dxf.start[1] - yMin) * escala),
                int((e.dxf.end[0] - xMin) * escala),
                int((e.dxf.end[1] - yMin) * escala)
            ]
            walls.append(line)

    return walls


@jit
def side(aX, aY, bX, bY, cX, cY):
    """
    Returns a position of the point c relative to the line going through a and b
        Points a, b are expected to be different.
    :param cY:
    :param cX:
    :param bY:
    :param bX:
    :param aY:
    :param aX:
    :return:
    """
    d = (cY - aY) * (bX - aX) - (bY - aY) * (cX - aX)
    return 1 if d > 0 else (-1 if d < 0 else 0)


@jit
def is_point_in_closed_segment(aX, aY, bX, bY, cX, cY):
    """
    Returns True if c is inside closed segment, False otherwise.
        a, b, c are expected to be collinear
    :param cY:
    :param cX:
    :param bY:
    :param bX:
    :param aY:
    :param aX:
    :return: Retorna valor booleano True se for um ponto fechado por segmento de reta. Caso contrario retorna False.
    """
    if aX < bX:
        return aX <= cX <= bX
    if bX < aX:
        return bX <= cX <= aX

    if aY < bY:
        return aY <= cY <= bY
    if bY < aY:
        return bY <= cY <= aY

    return aX == cX and aY == cY


@jit
def closed_segment_intersect(aX, aY, bX, bY, cX, cY, dX, dY):
    """
    Verifies if closed segments a, b, c, d do intersect.
    :param aX:
    :param aY:
    :param bX:
    :param bY:
    :param cX:
    :param cY:
    :param dX:
    :param dY:
    :return:
    """
    if (aX == bX) and (aY == bY):
        return (aX == cX and aY == cY) or (aX == dX and aY == dY)
    if (cX == dX) and (cY == dY):
        return (cX == aX and cY == aY) or (cX == bX and cY == bY)

    # TODO ao inves de invocar a funcao side, colocar a formula aqui
    s1 = side(aX, aY, bX, bY, cX, cY)
    s2 = side(aX, aY, bX, bY, dX, dY)

    # All points are collinear
    if s1 == 0 and s2 == 0:
        # TODO ao inves de invocar a funcao is_point_in_closed_segment, colocar a formula aqui
        return \
            is_point_in_closed_segment(aX, aY, bX, bY, cX, cY) or is_point_in_closed_segment(aX, aY, bX, bY, dX, dY) or \
            is_point_in_closed_segment(cX, cY, dX, dY, aX, aY) or is_point_in_closed_segment(cX, cY, dX, dY, bX, bY)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    s1 = side(cX, cY, dX, dY, aX, aY)
    s2 = side(cX, cY, dX, dY, bX, bY)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    return True


@jit
def absorption_in_walls(apX, apY, destinyX, destinyY, floor_plan):
    intersections = 0

    size = len(floor_plan)

    # if size > 0:
    #     intersections = 1000000000

    for i in range(size):
        # Coordenadas da parede

        if closed_segment_intersect(apX, apY, destinyX, destinyY, floor_plan[i][0], floor_plan[i][1], floor_plan[i][2],
                                    floor_plan[i][3]):
            intersections += 1

    intersecoes_com_paredes = intersections

    # parede de concredo, de 8 a 15 dB. Por conta da precisao em casas decimais do float32, é melhor pegar a ordem de
    # magnitude com o dBm do que tentar usar o valor exato com mW
    # dbm_absorvido_por_parede = 8 ## AGORA É UMA CONSTANTE GLOBAL

    return intersecoes_com_paredes * dbm_absorvido_por_parede


@jit
def mw_to_dbm(mW):
    """
    Método que converte a potência recebida dada em mW para dBm
    :param mW: Valor em miliwatts.
    :return: Valor de miliwatts convertido para decibéis.
    """
    return 10. * log10(mW)


@jit
def dbm_to_mw(dBm):
    """
    Método que converte a potência recebida dada em dBm para mW.
    :param dBm: Valor em decibéis.
    :return: Valor de decibéis convertidos em miliwatts.
    """
    return 10 ** (dBm / 10.)


@jit
def calc_distance(x1, y1, x2, y2):
    """
    Método responsável por realizar o calculo da distância entre dois pontos no plano cartesiano.
    :param x1: Valor de X no ponto 1.
    :param y1: Valor de Y no ponto 1.
    :param x2: Valor de X no ponto 2.
    :param y2: Valor de Y no ponto 2.
    :return: Retorna um valor float representando a distância dos pontos informados.
    """
    return sqrt(pow((x1 - x2), 2.0) + pow((y1 - y2), 2.0)) * precisao


@jit
def log_distance(d, gamma=3, d0=1, Pr_d0=-60, Pt=-17):
    """
       Modelo logaritmo de perda baseado em resultados experimentais. Independe da frequência do sinal transmitido
       e do ganho das antenas transmissora e receptora.
       Livro Comunicações em Fio - Pricipios e Práticas - Rappaport (páginas 91-92).
       :param Pr_d0:
       :param Pt:
       :param d0: Distância do ponto de referência d0.
       :param d: Distância que desejo calcular a perda do sinal.
       :param gamma: Valor da constante de propagação que difere para cada tipo de ambiente.
       :return: Retorna um float representando a perda do sinal entre a distância d0 e d.
       """

    # path_loss(d0) + 10 * gamma * log10(d / d0)
    # HAVIAMOS CODIFICADO ASSIM PARA ECONOMIZAR 1 SUBTRACAO e 1 VAR
    # return 17 - (60 + 10 * gamma * log10(d / d0))  # igual está na tabela

    # REESCREVI FACILITAR A COMPREENSAO
    # return   -( PL + 10 * gamma * log10(d / d0) )
    # return 0 - (PL + 10 * gamma * log10(d / d0) )
    # return   - (PL + 10 * gamma * log10(d / d0) )
    # return   -PL   - 10 * gamma * log10(d / d0)
    # return   -(Pt-Pr0)   - (10 * gamma * log10(d / d0))
    # return   -Pt + Pr0   - (10 * gamma * log10(d / d0))
    # return   Pr0  - 10 * gamma * log10(d / d0) - Pt
    return (Pr_d0 - 10 * gamma * log10(d / d0)) - Pt


@jit
def log_distance_v2(d, gamma=3, d0=10, Pr_d0=-69, Pt=-20):
    # return   -( PL + 10 * gamma * log10(d / d0) )
    return (Pr_d0 - 10 * gamma * log10(d / d0)) - Pt


@jit
def tree_par_log(x):
    return -17.74321 - 15.11596 * math.log(x + 2.1642)


@jit
def two_par_logistic(x):
    # https://en.wikipedia.org/wiki/Logistic_distribution#Related_distributions
    return Pt_dBm - (-15.11596 * math.log10(x * 2.1642))


@jit
def four_par_log(x):
    A = 79.500
    B = -38
    C = -100.000
    D = 0.0
    E = 0.005

    # https://en.wikipedia.org/wiki/Shifted_log-logistic_distribution
    return Pt_dBm - (D + (A - D) / (pow((1 + pow((x / C), B)), E)))


@jit
def five_par_log(x):
    A = 84.0
    B = -48
    C = -121.0
    D = -5.0
    E = 0.005
    # https://en.wikipedia.org/wiki/Shifted_log-logistic_distribution
    return Pt_dBm - (D + (A - D) / (pow((1 + pow((x / C), B)), E)))


@jit
def propagation_model(x, y, apX, apY, floor_plan):
    d = calc_distance(x, y, apX, apY)

    loss_in_wall = absorption_in_walls(apX, apY, x, y, floor_plan)

    if d == 0:
        d = 1

    # value = log_distance(d, 3, 11, -72, -20) - loss_in_wall
    # value = log_distance(d, 3, 1, -60, -17) - loss_in_wall
    # value = log_distance(d, 3, 10, -69, -20) - loss_in_wall
    # value = four_par_log(d) - loss_in_wall
    value = five_par_log(d) - loss_in_wall

    return value


@jit
def objective_function(matrix):
    # def objective_function(x):
    """
    Função objetivo para a avaliação da solução atual.
    :param matrix: Matriz a ser avaliada.
    :return: Retorna a soma de todos os elementos da metriz.
    """

    # TODO pra avaliar 2 FO de 2 APs, subtraia as duas matrizes (R[x][y] = abs(A[x][y]-B[x][y])) e pegue a soma de R
    # return abs(np.mean(matrix))

    # Desabilitado pois 'ficou pesado'.
    # minSensibilidade = dbm_to_mw(-84)
    # g = 0
    # for line in matrix:
    #     for value in line:
    #         g += -1/value
    #         # if value < minSensibilidade:
    #         #     g += -1
    #         # else:
    #         #     g += value
    #
    # return g
    # return abs(np.sum(np.power(10, matrix)))
    # return pow(10, x)

    # TODO: Penalizar os valores que estão abaixo da sensibilidade.
    return abs(np.sum(matrix))

    # sum_reduce = cuda.reduce(lambda a, b: a + b)
    # return sum_reduce(np.array([10 ** (x / 10.) for line in matrix for x in line]))


@cuda.jit
def objective_function_kernel(matrix, soma):
    """
    Função objetivo para a avaliação da solução atual.
    :param soma:
    :param matrix: Matriz a ser avaliada.
    :return: Retorna a soma de todos os elementos da metriz.
    """
    W = len(matrix)
    H = len(matrix[0])

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, W, gridX):
        for y in range(startY, H, gridY):
            soma += matrix[x][y]


@cuda.jit
def simulate_kernel(apX, apY, matrix_results, floor_plan):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param floor_plan:
    :param apY:
    :param apX:
    :param matrix_results:
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, WIDTH, gridX):
        for y in range(startY, HEIGHT, gridY):
            matrix_results[x][y] = propagation_model_gpu(x, y, apX, apY, floor_plan)


propagation_model_gpu = cuda.jit(device=True)(propagation_model)


@jit
def simulate_cpu(apX, apY, matrix_results, floor_plan):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param floor_plan:
    :param matrix_results:
    :param apY:
    :param apX:
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    for x in range(WIDTH):
        for y in range(HEIGHT):
            matrix_results[x][y] = propagation_model(x, y, apX, apY, floor_plan)

    return matrix_results


@jit
def get_point_in_circle(pointX, pointY, ray):
    """
    MÃ©todo por retorna um ponto ou conjunto de pontos dentro de um determinado raio de um ponto.
    :param pointY:
    :param pointX:
    :param ray: Valor do raio desejado.
    valores discretos para posiÃ§Ãµes da matriz.
    :return: Um ponto ou um conjunto de pontos do tipo float.
    """
    num = 1

    t = np.random.uniform(0.0, 2.0 * np.pi, num)
    r = ray * np.sqrt(np.random.uniform(0.0, 1.0, num))

    x = r * np.cos(t) + pointX
    y = r * np.sin(t) + pointY

    # Converte todos os valores negativos da lista em positivos

    x = round(abs(x[0]))
    y = round(abs(y[0]))

    return list([x, y])


@jit
def perturba_array(S_array, size):
    """
     Função que realiza uma perturbação na Solução S.
     Solução pode ser perturbada em um raio 'r' dentro do espaço de simulação.
    :param size:
    :param S_array:
    :return: Retorna um ponto dentro do raio informado.
    """
    novoS = np.empty([num_aps, 2], np.float32)

    for i in range(size):
        # Obtem um ponto aleatorio em um raio de X metros
        novoS[i] = get_point_in_circle(S_array[i][0], S_array[i][1], RAIO_PERTURBACAO)

    return novoS


@jit
def perturba(S):
    """
     Função que realiza uma perturbação na Solução S.
     Solução pode ser perturbada em um raio 'r' dentro do espaço de simulação.
    :param S: Ponto atual.
    :return: Retorna um ponto dentro do raio informado.
    """

    return get_point_in_circle(S[0], S[1], RAIO_PERTURBACAO)


@jit
def avalia_array(S_array, size):
    matrizes_propagacao = []
    for i in range(size):
        matrizes_propagacao.append(simula_propagacao(S_array[i][0], S_array[i][1]))

    # TODO: só pra testes, simples demais
    # fo_APs = 0
    # for i in range(size):
    #     fo_APs += objective_function(matrizes_propagacao[i])
    #
    # return fo_APs

    # simplesmente guloso
    # matriz_sobreposta = sobrepoe_solucoes_MAX(matrizes_propagacao, size)

    # penaliza APs muito proximos
    matriz_sobreposta = sobrepoe_solucoes_DIV_dBm(matrizes_propagacao, size)

    return objective_function(matriz_sobreposta), matrizes_propagacao


@jit
def sobrepoe_solucoes_MAX(propagation_array, size):
    max = propagation_array[0]
    for i in range(1, size):
        max = np.maximum(propagation_array[i], max)

    return max


@jit
def sobrepoe_solucoes_SUB(propagation_array, size):
    sub = propagation_array[0]
    for i in range(1, size):
        sub = np.subtract(propagation_array[i], sub)

    return sub


@jit
def sobrepoe_solucoes_DIV_dBm(propagation_array, size):
    # verificar se é veridico
    if size == 1:
        return propagation_array[0]

    matrixMin = propagation_array[0]
    matrixMax = propagation_array[0]

    for i in range(1, size):
        matrixMin = np.minimum(propagation_array[i], matrixMin)
        matrixMax = np.maximum(propagation_array[i], matrixMax)

    # pois ao subtrair dBm, deve ser o maior/menor
    sub = np.divide(matrixMax, matrixMin)

    return sub


@jit
def simula_propagacao_cpu(apX, apY):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param apY:
    :param apX:
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    matrix_results = np.empty([WIDTH, HEIGHT], np.float32)

    return simulate_cpu(apX, apY, matrix_results, floor_plan)


@jit
def simula_propagacao_gpu(pointX, pointY):
    """
    Valor da função objetivo correspondente á configuração x;
    :param pointX:
    :param pointY: Ponto para realizar a simulação.
    :return: Retorna um numero float representando o valor da situação atual.
    """
    g_matrix = np.zeros(shape=(WIDTH, HEIGHT), dtype=np.float32)

    blockDim = (48, 8)
    gridDim = (32, 16)

    d_matrix = cuda.to_device(g_matrix)

    simulate_kernel[gridDim, blockDim](pointX, pointY, d_matrix, floor_plan)

    d_matrix.to_host()

    return g_matrix


@jit
def simula_propagacao(pointX, pointY):
    """
    Método resposável por realizar a simulação da propagação de acordo com o ambiente escolhido (CPU ou GPU)
    :param pointX:
    :param pointY:
    :return:
    """

    if ENVIRONMENT == "GPU":
        # with GPU CUDA Threads
        return simula_propagacao_gpu(pointX, pointY)

    elif ENVIRONMENT == "CPU":
        #  with CPU Threads
        return simula_propagacao_cpu(pointX, pointY)
    else:
        exit(-1)

@jit
def objective_function_mW(array_matrix):

    matrix = sobrepoe_solucoes_MAX(array_matrix, len(array_matrix))

    sum = 0

    for line in matrix:
        for value in line:
            sum += dbm_to_mw(value)

    return sum


def simulated_annealing(size, M, P, L, T0, alpha):
    """
    :param size:
    :param T0: Temperatura inicial.
    :param M: Número máximo de iterações (Entrada).
    :param P: Número máximo de Perturbações por iteração (Entrada).
    :param L: Número máximo de sucessos por iteração (Entrada).
    :param alpha: Factor de redução da temperatura (Entrada).
    :return: Retorna um ponto sendo o mais indicado.
    """

    # cria Soluções iniciais com pontos aleatórios para os APs
    S_array = np.empty([size, 2], np.float32)

    for i in range(size):
        S_array[i] = [rd.randrange(0, WIDTH), rd.randrange(0, HEIGHT)]
        # S_array[i] = [WIDTH * 0.5, HEIGHT * 0.5]

    S0 = S_array.copy()
    print("Solução inicial:\n" + str(S0))

    result = avalia_array(S_array, size)
    fS = result[0]

    T = T0
    j = 1

    i_ap = 0

    # Armazena a MELHOR solução encontrada
    BEST_S_array = S_array.copy()
    BEST_fS = fS

    # Loop principal – Verifica se foram atendidas as condições de termino do algoritmo
    while True:
        i = 1
        nSucesso = 0

        # Loop Interno – Realização de perturbação em uma iteração
        while True:

            Si_array = S_array.copy()

            # a cada iteração do SA, perturba um dos APs
            i_ap = (i_ap + 1) % num_aps

            Si_array[i_ap] = perturba(S_array[i_ap])

            # retorna a FO e suas matrizes
            result = avalia_array(Si_array, num_aps)
            fSi = result[0]
            matrix_FO = result[1]

            ## Cuidado pois fica demasiado lento o desempenho do SA
            # if ANIMACAO_PASSO_A_PASSO:
            # 	show_solution(S_array, DISPLAYSURF)

            # Verificar se o retorno da função objetivo está correto. f(x) é a função objetivo
            deltaFi = fSi - fS

            # Minimização: deltaFi >= 0
            # Maximização: deltaFi <= 0
            # Teste de aceitação de uma nova solução
            if (deltaFi <= 0) or (exp(-deltaFi / T) > random()):

                S_array = Si_array.copy()

                fS = fSi
                nSucesso = nSucesso + 1

                if fS > BEST_fS:
                    BEST_fS = fS
                    BEST_S_array = S_array.copy()
                    BEST_matrix_FO = matrix_FO

                ## Cuidado pois fica demasiado lento o desempenho do SA
                # if ANIMACAO_MELHORES_LOCAIS:
                # 	show_solution(S_array, DISPLAYSURF)

                print("FO: " + '{:.3e}'.format(float(fS)))

                FOs.append(objective_function_mW(matrix_FO))

            i = i + 1

            if (nSucesso >= L) or (i > P):
                break

        # Atualização da temperatura (Deicaimento geométrico)
        T = alpha * T

        # Atualização do contador de iterações
        j = j + 1

        if (nSucesso == 0) or (j > M):
            break

    # saiu do loop principal
    # show_solution(S)
    # print("invocacoes de f(): " + str(contador_uso_func_objetivo))

    print("Distância da solução inicial:\t\t\t\t\t" + str(sobrepoe_solucoes_SUB(S_array, num_aps)))

    FOs.append(objective_function_mW(BEST_matrix_FO))  ## AQUI

    return BEST_S_array

def hex_to_rgb(hex):
    """
    Método responsável por converter uma cor no formato hexadecial para um RGB.
    :param hex: Valor em hexadecimal da cor.
    :return: Tupla representando a cor em formato RGB.
    """
    # hex = str(hex).lstrip('#')
    # return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))

    hex = str(hex).lstrip('#')
    lv = len(hex)
    return tuple(int(hex[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    # corR = int(Color(hex).get_red() * 255)
    # corG = int(Color(hex).get_green() * 255)
    # corB = int(Color(hex).get_blue() * 255)
    #
    # return tuple([corR, corG, corB])


def draw_line(DISPLAYSURF, x1, y1, x2, y2, color):
    """
    Método responsável por desenhar uma linha reta usando o PyGame de acordo com a posição de dois pontos.
    :param DISPLAYSURF:
    :param x1: Valor de X no ponto 1.
    :param y1: Valor de Y no ponto 1.
    :param x2: Valor de X no ponto 2.
    :param y2: Valor de Y no ponto 2.
    :param color: Cor que a linha irá ter.
    :return: None
    """
    pygame.draw.line(DISPLAYSURF, color, (x1, y1), (x2, y2))


def print_pygame_pyOpenGL(matrix_results, access_points, DISPLAYSURF):
    # pxarray = pygame.PixelArray (surface)
    x = 0


def print_pygame(matrix_results, access_points, DISPLAYSURF):
    """
    Método responsável por desenhar a simulação usando o PyGame.
    :param DISPLAYSURF:
    :param access_points:
    :param matrix_results: Matriz float contendo os resultados da simulação.
    :return: None.
    """

    matrix_max_value = matrix_results.max()
    # #matrix_min_value = matrix_results.min()

    # # Se utilizar a função min tradicional, a penalização de DBM_MIN_VALUE irá interferir no range de cor
    # matrix_min_value = matrix_max_value
    # for x in range(WIDTH):
    #     for y in range(HEIGHT):
    #         if matrix_results[x][y] != DBM_MIN_VALUE and matrix_results[x][y] < matrix_min_value:
    #             matrix_min_value = matrix_results[x][y]

    # matrix_max_value = -30
    matrix_min_value = -100

    # print("Desenhando simulação com PyGame...")

    # Lê os valores da matriz que contêm valores calculados e colore
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = get_color_of_interval(matrix_results[x][y], matrix_max_value, matrix_min_value)
            draw_point(DISPLAYSURF, color, x, y)

    # Pinta de vermelho a posição dos Access Points
    for ap in access_points:
        draw_point(DISPLAYSURF, RED, ap[0], ap[1])

    # draw_floor_plan(floor_plan)

    # Atualiza a janela do PyGame para que exiba a imagem
    pygame.display.update()


def draw_point(DISPLAYSURF, color, x, y):
    """
    Método responsável por desenhar um ponto usando o PyGame de acordo com a posição (x,y).
    :param DISPLAYSURF:
    :param color: A cor que irá ser o ponto.
    :param x: Posição do ponto no eixo X.
    :param y: Posição do ponto no eixo Y.
    :return: None.
    """
    pygame.draw.line(DISPLAYSURF, color, (x, y), (x, y))


def size_of_floor_plan(floor_plan):
    """
    Método responsável por obter as dimenções da planta
    :param floor_plan:
    :return:
    """
    xMax = yMax = 0

    for lines in floor_plan:
        if lines[0] > xMax:
            xMax = lines[0]
        if lines[2] > xMax:
            xMax = lines[2]

        if lines[1] > yMax:
            yMax = lines[1]
        if lines[3] > yMax:
            yMax = lines[3]

    return [xMax, yMax]


def draw_floor_plan(floor_plan, DISPLAYSURF):
    for line in floor_plan:
        # draw_line(line[0]*escala, line[1]*escala, line[2]*escala, line[3]*escala, WHITE)
        draw_line(DISPLAYSURF, line[0], line[1], line[2], line[3], WHITE)

    # Atualiza a janela do PyGame para que exiba a imagem
    pygame.display.update()


def get_percentage_of_range(min, max, x):
    """
    Método responsável por retornar a porcentagem de acordo com um respectivo intervalo.
    :param min: Valor mínimo do intervalo.
    :param max: Valor máximo do intervalo.
    :param x: Valor que está no intervalo de min-max que deseja saber sua respectiva porcentagem.
    :return: Retorna uma porcentagem que está de acordo com o intervalo min-max.
    """

    return ((x - min) / (max - min)) * 100


def get_value_in_list(percent, list):
    """
    Método retorna o valor de uma posição de uma lista. A posição é calculada de acordo a porcentagem.
    :param percent: Valor float representando a porcentagem.
    :param list: Lista com n números.
    :return: Retorna a cor da posição calculada.
    """
    position = (percent / 100) * len(list)
    if position < 1:
        position = 1
    elif position >= len(list):
        position = len(list)
    return hex_to_rgb(list[int(position - 1)])
    # return list[int(position - 1)]


def get_color_of_interval(x, max=-30, min=-100):
    """
    Este método retorna uma cor de acordo com o valor que está entre o intervalo min-max. Em outras palavras,
    este método transforma um número em uma cor dentro de uma faixa informada.
    :param min: Valor mínimo do intervalo.
    :param max: Valor máximo do intervalo.
    :param x: Valor que está dentro do intervalo e que deseja saber sua cor.
    :return: Retorna uma tupla representando um cor no formato RGB.
    """

    if PAINT_BLACK_BELOW_SENSITIVITY and x < SENSITIVITY:
        return BLACK

    percentage = get_percentage_of_range(min, max, x)
    color = get_value_in_list(percentage, COLORS)

    return color


def show_solution_opengl(S_array):
    # print("\nDesenhando resultado da simulação com PyOpenGL.")

    matrizes_propagacao = []
    for i in range(len(S_array)):
        matrizes_propagacao.append(simula_propagacao(S_array[i][0], S_array[i][1]))
    # propagacao = sobrepoe_solucoes_ADD(matrizes_propagacao, len(S_array))
    propagacao = sobrepoe_solucoes_MAX(matrizes_propagacao, len(S_array))

    print_pygame(propagacao, S_array, DISPLAYSURF)
    draw_floor_plan(walls, DISPLAYSURF)


def show_solution(S_array, DISPLAYSURF):
    print("\nDesenhando resultado da simulação com PyGame.")

    matrizes_propagacao = []

    for i in range(len(S_array)):
        matrizes_propagacao.append(simula_propagacao(S_array[i][0], S_array[i][1]))

    # propagacao = sobrepoe_solucoes_ADD(matrizes_propagacao, len(S_array))
    propagacao = sobrepoe_solucoes_MAX(matrizes_propagacao, len(S_array))

    print_pygame(propagacao, S_array, DISPLAYSURF)

    draw_floor_plan(walls, DISPLAYSURF)


def get_color_gradient(steps=250):
    cores = list(Color("red").range_to(Color("green"), steps))
    # cores = list(Color("blue").range_to(Color("red"), steps))
    cores.pop(0)
    cores.pop(len(cores) - 1)

    return cores


def show_configs():
    print("\nOtimização via Simulated Annealing com a seguinte configuração:" + "\n")
    print("\tNúmeto máximo de iterações:\t\t\t" + str(max_inter))
    print("\tNúmero máximo de pertubações por iteração:\t" + str(max_pertub))
    print("\tNúmero máximo de sucessos por iteração:\t\t" + str(num_max_succ))
    print("\tTemperatura inicial:\t\t\t\t" + str(temp_inicial))
    print("\tDecaimento da teperatura com α=\t\t\t" + str(alpha))
    print("\tRaio de perturbação:\t\t\t\t" + str(RAIO_PERTURBACAO))
    print("\nHardware de simulação:\t\t\t\t" + str(ENVIRONMENT) + "\n")

    print("\nSimulação do ambiente com a seguinte configuração:" + "\n")
    print("\tQuantidade de soluções finais:\t\t" + str(max_SA))
    print("\tSimulando ambiente com :\t\t\t" + str(WIDTH) + "x" + str(HEIGHT) + " pixels")
    print("\tEscala de simulação da planta:\t\t\t1 px : " + str(1 / escala) + " metros")


def run():
    # variasSolucoes = []
    #
    # for i in range(max_SA):
    #     print("Calculando o melhor ponto [" + str(i) + "]")
    #     variasSolucoes.append(
    #         simulated_annealing(num_aps, max_inter, max_pertub, num_max_succ, temp_inicial, alpha))
    #
    # maxFO = 0
    # bestSolution = variasSolucoes[0]
    #
    # print("Analizando a melhor solução.")
    #
    # for ap_array in variasSolucoes:
    #     ap_array_fo = avalia_array(ap_array)
    #     if ap_array_fo > maxFO:
    #         maxFO = ap_array_fo
    #         bestSolution = ap_array

    #    # Visualização dos dados
    # # Inicia o PyGame e configura o tamanho da janela
    #    pygame.init()
    #    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

    bestSolution = simulated_annealing(num_aps, max_inter, max_pertub, num_max_succ, temp_inicial, alpha)
    result = avalia_array(bestSolution, len(bestSolution))
    bestSolution_fo = result[0]

    # print("\nMelhor ponto sugerido pelo algoritmo: " + str(bestSolution) + "\n FO: " + str(bestSolution_fo))
    print("\nMelhor ponto sugerido pelo algoritmo: " + str(bestSolution) + "\n FO: " + '{:.3e}'.format(
        float(bestSolution_fo)))

    print("\nDesenhando resultado da simulação...")
    show_solution(bestSolution, DISPLAYSURF)
    # show_solution(1, 1)

    # Gera resumo da simulação
    generate_summary(bestSolution)

def test_propagation():
    """
    Método usado apenas para fim de testes com a simulação em pontos específicos.
    :return: None.
    """
    test_AP_in_the_middle = [[int(WIDTH / 2), int(HEIGHT / 2)]]

    # Inicia o PyGame
    pygame.init()

    # Configura o tamanho da janela
    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    #
    show_solution(test_AP_in_the_middle, DISPLAYSURF)
    # show_solution(1, 1)


def generate_summary(S_array):
    length = len(S_array)

    print("\n****** Gerando sumários dos resultados da simulação ******")
    print("Numero de soluções:\t" + str(length))

    for i in range(length):

        print("\nAvaliando solução (" + str(i + 1) + "/" + str(length) + ")\t\tPonto:\t(" + str(
            S_array[i][0]) + "," + str(S_array[i][1]) + ")")

        matrix = simula_propagacao(S_array[i][0], S_array[i][1])

        above_sensitivity = [value for line in matrix for value in line if value > SENSITIVITY]
        between_sensitivity = [value for line in matrix for value in line if value == SENSITIVITY]
        under_sensitivity = [value for line in matrix for value in line if value < SENSITIVITY]

        total = WIDTH * HEIGHT

        percent_cover_above_sensitivity = (len(above_sensitivity) / total) * 100
        percent_cover_between_sensitivity = (len(between_sensitivity) / total) * 100
        percent_cover_under_sensitivity = (len(under_sensitivity) / total) * 100

        print(
            "\t" + str(round(percent_cover_above_sensitivity, 2)) + "%\tdos pontos estão acima da sensibilidade do AP.")
        print("\t" + str(round(percent_cover_between_sensitivity, 2)) + "%\tdos pontos estão sob sensibilidade do AP.")
        print("\t" + str(
            round(percent_cover_under_sensitivity, 2)) + "%\tdos pontos estão abaixo da sensibilidade do AP.")

        faixa1 = faixa2 = faixa3 = faixa4 = faixa5 = 0

        #   0 a -29 dBm -> faixa1
        # -30 a -49 dBm -> faixa2
        # -50 a -69 dBm -> faixa3
        # -70 a -85 dBm -> faixa4
        # -86 a -100 dBm -> faixa5

        for line in matrix:
            for value in line:

                if value > 0 or value > -29:
                    faixa1 += 1

                elif -30 > value > -49:
                    faixa2 += 1

                elif -50 > value > -69:
                    faixa3 += 1

                elif -70 > value > -85:
                    faixa4 += 1

                elif value < -86 or value < -100:
                    faixa5 += 1

        percent_faixa1 = faixa1 / total * 100
        percent_faixa2 = faixa2 / total * 100
        percent_faixa3 = faixa3 / total * 100
        percent_faixa4 = faixa4 / total * 100
        percent_faixa5 = faixa5 / total * 100

        print("\n\tFaixa\t\t\tCobertura")
        print("\t< 0 ~ -29 dBm\t\t" + str(round(percent_faixa1, 2)) + "%")
        print("\t-30 ~ -49 dBm\t\t" + str(round(percent_faixa2, 2)) + "%")
        print("\t-50 ~ -69 dBm\t\t" + str(round(percent_faixa3, 2)) + "%")
        print("\t-70 ~ -85 dBm\t\t" + str(round(percent_faixa4, 2)) + "%")
        print("\t-86 ~ -100 dBm >\t" + str(round(percent_faixa5, 2)) + "%")

    print("Gerando gráfico do comportamento da FO.")

    # Plota gráfico da função objetivo
    plt.plot(FOs)
    plt.title("Comportamento da FO.")
    plt.ylabel('Valor FO')
    plt.xlabel('Quantidade de pontos')
    plt.show()


########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################
if __name__ == '__main__':
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    ##################################################
    #  CONFIGURAÇÕES DOS EQUIPAMENTOS

    # OBS.: por conta da precisao de casas decimais do float
    #        é melhor pegar a ordem de magnitude com o dBm do
    #        que tentar usar o valor exato com mW

    # Potência de transmissão de cada AP
    Pt_dBm = -20

    # Sensibilidade dos equipamentos receptores
    SENSITIVITY = -85

    # Gradiente de cores da visualização gráfica
    COLORS = get_color_gradient(16)  # 64, 32, 24, 16, 8

    PAINT_BLACK_BELOW_SENSITIVITY = True
    # PAINT_BLACK_BELOW_SENSITIVITY = False

    DBM_MIN_VALUE = np.finfo(np.float32).min

    # parede de concredo, de 8 a 15 dB.
    dbm_absorvido_por_parede = 8

    ##################################################
    #  CONFIGURAÇÕES DO AMBIENTE E PLANTA-BAIXA

    COMPRIMENTO_BLOCO_A = 48.0
    COMPRIMENTO_BLOCO_B = 36.0
    COMPRIMENTO_BLOCO_C = 51.0

    COMPRIMENTO_EDIFICIO = COMPRIMENTO_BLOCO_B
    # LARGURA_EDIFICIO = ???

    # dxf_path = "./DXFs/bloco_a/bloco_A_planta baixa_piso1.dxf"
    # dxf_path = "./DXFs/bloco_a/bloco_A_planta baixa_piso1_porta.dxf"

    # dxf_path = "./DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso1.dxf"
    dxf_path = "./DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso2.dxf"
    # dxf_path = "./DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso3.dxf"
    # dxf_path = "./DXFs/bloco_c/sem_porta/bloco_C_planta_baixa_piso1.dxf"
    # dxf_path = "./DXFs/bloco_c/sem_porta/bloco_C_planta baixa_piso2.dxf"
    # dxf_path = "./DXFs/bloco_c/sem_porta/bloco_C_planta baixa_piso3.dxf"

    # carrega para saber o comprimento da planta
    walls = read_walls_from_dxf(dxf_path, 1)
    floor_plan = np.array(walls, dtype=np.float32)

    floor_size = size_of_floor_plan(walls)
    comprimento_planta = floor_size[0]
    largura_planta = floor_size[1]

    ##################################################
    #  CONFIGURAÇÕES DO AMBIENTE SIMULADO

    ENVIRONMENT = "GPU"
    # ENVIRONMENT = "CPU"

    # Tamanho da simulação
    TAMAMHO_SIMULACAO = 400

    # Ativa / Desativa a animação passo a passo da otimização
    # ANIMACAO_PASSO_A_PASSO   = True
    ANIMACAO_PASSO_A_PASSO = False

    # ANIMACAO_MELHORES_LOCAIS = True
    ANIMACAO_MELHORES_LOCAIS = False

    # Quantidade de APs
    num_aps = 2

    ##################################################

    # Lista para guardar as funções objetivos calculadas durante a simulação
    FOs = []

    WIDTH = TAMAMHO_SIMULACAO
    HEIGHT = int(WIDTH * (largura_planta / comprimento_planta))
    escala = WIDTH / comprimento_planta
    precisao = COMPRIMENTO_EDIFICIO / WIDTH

    # HEIGHT = TAMAMHO_SIMULACAO
    # WIDTH = int(HEIGHT * (comprimento_planta / largura_planta))
    # escala = HEIGHT / largura_planta
    # precisao = LARGURA_EDIFICIO / WIDTH

    # RE-carrega utilizando a escala apropriada
    walls = read_walls_from_dxf(dxf_path, escala)
    floor_plan = np.array(walls, dtype=np.float32)
    ##################################################

    ##################################################
    #  CONFIGURAÇÕES DO OTIMIZADOR

    # fixo, procurar uma fórmula para definir o max_iter em função do tamanho da matriz (W*H)
    # max_inter = 600 * (1 + num_aps)
    max_inter = 600 * (10 * num_aps)

    # p - Máximo de perturbações
    max_pertub = 5

    # RAIO_PERTURBACAO = WIDTH * 0.01
    # RAIO_PERTURBACAO = WIDTH * 0.0175
    # RAIO_PERTURBACAO = WIDTH * 0.025
    # RAIO_PERTURBACAO = WIDTH * 0.11
    RAIO_PERTURBACAO = WIDTH * 0.025 * num_aps

    # v - Máximo de vizinhos
    # num_max_succ = 80
    num_max_succ = 80 * 10

    # a - Alpha
    alpha = .85
    # alpha = .95

    # t - Temperatura
    # temp_inicial = 300 * (1 + num_aps)
    temp_inicial = 300 * (1 + num_aps) * 10

    # Máximo de iterações do S.A.
    max_SA = 1
    ##################################################

    # Visualização dos dados
    # Inicia o PyGame e configura o tamanho da janela
    pygame.init()
    icon = pygame.image.load('images/icon.png')
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Resultado Simulação - IFMG Campus Formiga")
    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

    show_configs()
    # test_propagation()
    run()

    # profile.runctx('run()', globals(), locals(),'tese')
    # cProfile.run(statement='run()', filename='PlacementAPs.cprof')

    # python ../PlacementAPs.py | egrep "(tottime)|(PlacementAPs.py)" | tee ../cProfile/PlacementAPs.py_COM-JIT.txt
    # cat ../cProfile/PlacementAPs.py_COM-JIT.txt | sort -k 2 -r

    # python PlacementAPs.py | egrep '(ncalls)|(PlacementAPs)'
    # https://julien.danjou.info/blog/2015/guide-to-python-profiling-cprofile-concrete-case-carbonara

    # generate_summary([[50, 50]])

    input('\nAperte ESC para fechar a simulação.')

    # profile.runctx('run()', globals(), locals(),'tese')
    # cProfile.run(statement='run()', filename='PlacementAPs.cprof')

    # python ../PlacementAPs.py | egrep "(tottime)|(PlacementAPs.py)" | tee ../cProfile/PlacementAPs.py_COM-JIT.txt
    # cat ../cProfile/PlacementAPs.py_COM-JIT.txt | sort -k 2 -r

    # python PlacementAPs.py | egrep '(ncalls)|(PlacementAPs)'
    # https://julien.danjou.info/blog/2015/guide-to-python-profiling-cprofile-concrete-case-carbonara

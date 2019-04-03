#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import profile
import random as rd
from math import sqrt, log10, exp
from random import random

import ezdxf
import numpy as np
import pygame
from colour import Color
from numba import cuda, jit

import cProfile

"""
Algoritmo que realiza a simulação da propagação do sinal wireless de determinado ambiente 2D de acordo com um Access
Point Informado.
"""


@jit
def read_walls_from_dxf(dxf_path):
    """
    Método responsável por ler um arquivo DXF e filtrar pela camada ARQ as paredes do ambiente.
    :param dxf_path: Caminho do arquivo de entrada, sendo ele no formato DFX.
    :return: Retorna uma lista contendo em cada posição, uma lista de quatro elementos, sendo os dois primeiros
    referêntes ao ponto inicial da parede e os dois ultimo referênte ao ponto final da parede.
    """
    dwg = ezdxf.readfile(dxf_path)

    walls = []

    modelspace = dwg.modelspace()

    ##TODO 7 por que?
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
    :param a: Ponto A.
    :param b: Ponto B.
    :param c: Ponto C.
    :return:
    """
    d = (cY - aY) * (bX - aX) - (bY - aY) * (cX - aX)
    return 1 if d > 0 else (-1 if d < 0 else 0)


@jit
def is_point_in_closed_segment(aX, aY, bX, bY, cX, cY):
    """
    Returns True if c is inside closed segment, False otherwise.
        a, b, c are expected to be collinear
    :param a: Ponto A.
    :param b: Ponto B.
    :param c: Ponto C.
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
    """ Verifies if closed segments a, b, c, d do intersect.
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


## TODO: otimizar este procedimento pois está fazendo a simulação ficar 163x mais lento
## @numba.jit("float32( int32[2], int32[2], List(List(int64)) )", target='parallel')
## @numba.jit(target='cpu', forceobj=True)
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

    ## OBS.: dividir por dois se cada parede for um retangulo no DXF
    # intersecoes_com_paredes = intersections / 2
    intersecoes_com_paredes = intersections

    # parede de concredo, de 8 a 15 dB. Por conta da precisao em casas decimais do float32, é melhor pegar a ordem de
    # magnitude com o dBm do que tentar usar o valor exato com mW
    dbm_absorvido_por_parede = 8

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

    ## path_loss(d0) + 10 * gamma * log10(d / d0)
    ## HAVIAMOS CODIFICADO ASSIM PARA ECONOMIZAR 1 SUBTRACAO e 1 VAR
    # return 17 - (60 + 10 * gamma * log10(d / d0))  # igual está na tabela

    ## REESCREVI FACILITAR A COMPREENSAO
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

    ##TODO pra avaliar 2 FO de 2 APs, subtraia as duas matrizes (R[x][y] = abs(A[x][y]-B[x][y])) e pegue a soma de R
    # return abs(np.mean(matrix))

    ## Desabilitado pois 'ficou pesado'.
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

    ## TODO: Penalizar os valores que estão abaixo da sensibilidade.
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


def simulate_cpu(apX, apY, floor_plan):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param floor_plan:
    :param apY:
    :param apX:
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    matrix_results = np.zeros(shape=(WIDTH, HEIGHT))

    for x in range(WIDTH):
        for y in range(HEIGHT):
            value = propagation_model(x, y, apX, apY, floor_plan)
            matrix_results[x][y] = value

    return matrix_results


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


# TODO copiar a versão mais enxuta do PlacementGPUaps.py

# def get_point_in_circle(pointX, pointY, ray, round_values=True, num=1, absolute_values=True):
#     """
#     Método por retorna um ponto ou conjunto de pontos dentro de um determinado raio de um ponto.
#     :param point: Ponto contendo posição [x, y] de referência do ponto.
#     :param ray: Valor do raio desejado.
#     :param round_values: Flag que informa se o(s) ponto(s) serão arredondados. Geralmente será usando para retornar
#     valores discretos para posições da matriz.
#     :param absolute_values: Flag que informa se o(s) ponto(s) serão absolutos (positivos).
#     :param num: Número de pontos que deseja gerar. Gera um ponto como default.
#     :param debug: Flag que quando informada True, printa na tela o(s) ponto(s) gerados e a distância do ponto de
#     referência.
#     :return: Um ponto ou um conjunto de pontos do tipo float.
#     """
#
#     t = np.random.uniform(0.0, 2.0 * np.pi, num)
#     r = ray * np.sqrt(np.random.uniform(0.0, 1.0, num))
#
#     x = r * np.cos(t) + pointX
#     y = r * np.sin(t) + pointY
#
#     # Converte todos os valores negativos da lista em positivos
#     if absolute_values:
#         x = [abs(k) for k in x]
#         y = [abs(k) for k in y]
#
#     if round_values:
#         x = [round(k) for k in x]
#         y = [round(k) for k in y]
#
#     # Verifica se o retorno será um ponto único ou uma lista de pontos.
#     if num == 1:
#         return [x[0], y[0]]
#     else:
#         return [x, y]

@jit
def get_point_in_circle(pointX, pointY, ray):
    """
    MÃ©todo por retorna um ponto ou conjunto de pontos dentro de um determinado raio de um ponto.
    :param point: Ponto contendo posiÃ§Ã£o [x, y] de referÃªncia do ponto.
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

    ## TODO: só pra testes, simples demais
    # fo_APs = 0
    # for i in range(size):
    #     fo_APs += objective_function(matrizes_propagacao[i])
    #
    # return fo_APs

    ## simplesmente guloso
    # matriz_sobreposta = sobrepoe_solucoes_MAX(matrizes_propagacao, size)

    ## penaliza APs muito proximos
    matriz_sobreposta = sobrepoe_solucoes_DIV_dBm(matrizes_propagacao, size)

    return objective_function(matriz_sobreposta)


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

    ## pois ao subtrair dBm, deve ser o maior/menor
    sub = np.divide(matrixMax, matrixMin)

    return sub


@jit
def simula_propagacao(pointX, pointY):
    """
    Valor da função objetivo correspondente á configuração x;
    :param pointX:
    :param pointY: Ponto para realizar a simulação.
    :return: Retorna um numero float representando o valor da situação atual.
    """

    if ENVIRONMENT == "GPU":

        g_matrix = np.zeros(shape=(WIDTH, HEIGHT), dtype=np.float32)

        blockDim = (48, 8)
        gridDim = (32, 16)

        d_matrix = cuda.to_device(g_matrix)

        simulate_kernel[gridDim, blockDim](pointX, pointY, d_matrix, floor_plan)

        d_matrix.to_host()

        # ----------------------------------------------------------------------------

        # g_matrix = np.asmatrix(g_matrix, dtype=np.float32)
        # g_soma = np.zeros(shape=(WIDTH, HEIGHT), dtype=np.float32)
        #
        # d_matrix = cuda.to_device(g_matrix)
        # d_soma = cuda.to_device(g_soma)
        #
        # objective_function_kernel[gridDim, blockDim](d_matrix, d_soma)
        #
        # d_matrix.to_host()
        # d_soma.to_host()
        #
        # return abs(np.sum(g_soma))
        # return objective_function(g_matrix)

        return g_matrix
    elif ENVIRONMENT == "CPU":
        return simulate_cpu(pointX, pointY, floor_plan)

    else:
        exit(-1)


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
        # S_array[i] = [rd.randrange(0, WIDTH), rd.randrange(0, HEIGHT)]
        S_array[i] = [WIDTH * 0.5, HEIGHT * 0.5]

    S0 = S_array.copy()
    print("Solução inicial:\n" + str(S0))

    fS = avalia_array(S_array, size)

    T = T0
    j = 1

    i_ap = 0

    # Loop principal – Verifica se foram atendidas as condições de termino do algoritmo
    while True:
        i = 1
        nSucesso = 0

        # Loop Interno – Realização de perturbação em uma iteração
        while True:

            # Tera que mandar o ponto atual e a matriz (certeza?) tbm. Realiza a seleção do ponto.
            # Si = perturb(S[0], S[1])
            # fSi = f(Si[0], Si[1])

            # TODO perturbar todos
            # Si_array = perturba_array(S_array, num_aps)
            Si_array = S_array.copy()

            # a cada iteração do SA, perturba um dos APs
            i_ap = (i_ap + 1) % num_aps

            Si_array[i_ap] = perturba(S_array[i_ap])

            fSi = avalia_array(Si_array, num_aps)

            # show_solution(Si) print("[\t" + (str(round((100 - 100 * fSi / fS) * 100, 1))) + "\t] S: " + str(S_array)
            #  + "\t Si: " + str(Si_array))

            # Verificar se o retorno da função objetivo está correto. f(x) é a função objetivo
            deltaFi = fSi - fS

            # print("deltaFi: " + str(deltaFi))

            # Minimização: deltaFi >= 0
            # Maximização: deltaFi <= 0
            # Teste de aceitação de uma nova solução
            if (deltaFi <= 0) or (exp(-deltaFi / T) > random()):  # randomize()):
                # print("Ponto escolhido: " + str(Si))
                # LEMBRETE: guardar o ponto anterior, S_prev = S (para ver o caminho do Si pro S_prev)
                S_array = Si_array
                fS = fSi
                nSucesso = nSucesso + 1

                # show_solution(S)
                # print("melhor S: " + str(S))

            i = i + 1

            if (nSucesso >= L) or (i > P):
                break

        # print("iteração: " + str(j))
        # print("temperat: " + str(T) + "\n")

        # Atualização da temperatura (Deicaimento geométrico)
        T = alpha * T

        # Atualização do contador de iterações
        j = j + 1

        if (nSucesso == 0) or (j > M):
            break

    ## saiu do loop principal
    # show_solution(S)
    # print("invocacoes de f(): " + str(contador_uso_func_objetivo))

    print("Distância da solução inicial:\t\t\t\t\t" + str(sobrepoe_solucoes_SUB(S_array, num_aps)))

    return S_array


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
    :param x1: Valor de X no ponto 1.
    :param y1: Valor de Y no ponto 1.
    :param x2: Valor de X no ponto 2.
    :param y2: Valor de Y no ponto 2.
    :param color: Cor que a linha irá ter.
    :return: None
    """
    pygame.draw.line(DISPLAYSURF, color, (x1, y1), (x2, y2))


def print_pygame(matrix_results, access_points, DISPLAYSURF):
    """
    Método responsável por desenhar a simulação usando o PyGame.
    :param DISPLAYSURF:
    :param matrix_results: Matriz float contendo os resultados da simulação.
    :param access_points: Posição (x, y) do ponto de acesso.
    :return: None.
    """
    # matrix_max_value = matrix_results.max()
    # matrix_min_value = matrix_results.min()

    # testes
    matrix_max_value = -100
    matrix_min_value = -10

    # Se utilizar a função min tradicionar, a penalização de DBM_MIN_VALUE irá interferir no range de cor
    # matrix_min_value = matrix_max_value
    # for x in range(WIDTH):
    #     for y in range(HEIGHT):
    #         if matrix_results[x][y] != DBM_MIN_VALUE and matrix_results[x][y] < matrix_min_value:
    #             matrix_min_value = matrix_results[x][y]

                # print("Desenhando simulação com PyGame...")

    # Lê os valores da matriz que contêm valores calculados e colore
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = get_color_of_interval(matrix_min_value, matrix_max_value, matrix_results[x][y])
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

    ##TODO escala de cor linear, mas poderia ser exponencial (logaritmica)
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


def get_color_of_interval(min, max, x):
    """
    Este método retorna uma cor de acordo com o valor que está entre o intervalo min-max. Em outras palavras,
    este método transforma um número em uma cor dentro de uma faixa informada.
    :param min: Valor mínimo do intervalo.
    :param max: Valor máximo do intervalo.
    :param x: Valor que está dentro do intervalo e que deseja saber sua cor.
    :return: Retorna uma tupla representando um cor no formato RGB.
    """

    # if x < SENSITIVITY:
    #     return hex_to_rgb("#000000")

    percentage = get_percentage_of_range(min, max, x)
    color = get_value_in_list(percentage, COLORS)

    return color


def showSolution(S_array, DISPLAYSURF):
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


def run():
    print("\nIniciando simulação com simulated Annealing com a seguinte configuração:")
    print("Númeto máximo de iterações:\t\t\t" + str(max_inter))
    print("Número máximo de pertubações por iteração:\t" + str(max_pertub))
    print("Número máximo de sucessos por iteração:\t\t" + str(num_max_succ))
    print("Temperatura inicial:\t\t\t\t" + str(temp_inicial))
    print("Decaimento da teperatura com α=\t\t\t" + str(alpha))
    print("Repetições do Simulated Annealing:\t\t" + str(max_SA) + "\n")

    print("Raio de perturbação:\t\t\t\t" + str(RAIO_PERTURBACAO))
    print("Simulando ambiente com :\t\t\t" + str(WIDTH) + "x" + str(HEIGHT) + " pixels")
    print("Escala de simulação da planta:\t\t\t1 px : " + str(escala) + " metros")
    print("Ambiente de simulação:\t\t\t\t" + str(ENVIRONMENT) + "\n")

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

    # bestSolution = simulated_annealing(num_aps, max_inter, max_pertub, num_max_succ, temp_inicial, alpha)
    # bestSolution_fo = avalia_array(bestSolution, len(bestSolution))

    # x = 1230
    # y = 360
    # xx = (1610/864)*660
    # yy = (600/435)*260

    xx = 1175
    yy = 360

    bestSolution = [[xx, yy]]

    # print("\nMelhor ponto sugerido pelo algoritmo: " + str(bestSolution) + "\n FO: " + str(bestSolution_fo))
    #
    # # Inicia o PyGame
    pygame.init()
    #
    # # Configura o tamanho da janela
    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    #
    showSolution(bestSolution, DISPLAYSURF)
    # # show_solution(1, 1)
    #
    input('\nFim de execução.')


########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################
if __name__ == '__main__':
    COLORS = get_color_gradient(25)

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    Pt_dBm = -20

    # ENVIRONMENT = "GPU"
    ENVIRONMENT = "CPU"

    # tamanho da matriz = dimensão da planta / precisão

    # dxf_path = "../DXFs/bloco_a/bloco_A_planta baixa_piso1.dxf"
    # dxf_path = "../DXFs/bloco_a/bloco_A_planta baixa_piso1_porta.dxf"

    # dxf_path = "../DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso1.dxf"
    dxf_path = "../DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso2.dxf"
    # dxf_path = "../DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso3.dxf"

    # dxf_path = "../DXFs/bloco_c/sem_porta/bloco_C_planta_baixa_piso1.dxf"
    # dxf_path = "../DXFs/bloco_c/sem_porta/bloco_C_planta baixa_piso2.dxf"
    # dxf_path = "../DXFs/bloco_c/sem_porta/bloco_C_planta baixa_piso3.dxf"

    escala = 1
    # walls = read_walls_from_dxf("./DXFs/bloco-A-l.dxf")
    walls = read_walls_from_dxf(dxf_path)
    floor_plan = np.array(walls, dtype=np.float32)

    floor_size = size_of_floor_plan(walls)
    comprimento_planta = floor_size[0]
    largura_planta = floor_size[1]
    ## carreguei a planta so para obter a proporcao
    proporcao_planta = comprimento_planta / largura_planta

    # HEIGHT = int(largura_planta)
    # WIDTH = int(comprimento_planta)
    HEIGHT = 600  # 40
    WIDTH = int(HEIGHT * proporcao_planta)

    escala = HEIGHT / largura_planta
    # escala = WIDTH / comprimento_planta
    # precisao = 1  # metro
    precisao = 36.0 / WIDTH

    # walls = read_walls_from_dxf("/home/samuel/PycharmProjects/TCC/DXFs/bloco-a-linhas-sem-porta.dxf")
    walls = read_walls_from_dxf(dxf_path)
    floor_plan = np.array(walls, dtype=np.float32)

    SENSITIVITY = -90
    DBM_MIN_VALUE = np.finfo(np.float32).min

    ## Quantidade de APs
    num_aps = 1

    ## fixo, procurar uma fórmula para definir o max_iter em função do tamanho da matriz (W*H)
    max_inter = 600 * num_aps

    ## p
    max_pertub = 5

    # RAIO_PERTURBACAO = WIDTH * 0.01
    # RAIO_PERTURBACAO = WIDTH * 0.0175
    RAIO_PERTURBACAO = WIDTH * 0.025

    ## v
    num_max_succ = 80

    ## a
    alpha = .85

    ## t
    temp_inicial = 300 * 2

    ## máximo de iterações do S.A.
    max_SA = 1

    # run()
    # profile.runctx('run()', globals(), locals(),'tese')
    cProfile.run(statement='run()', filename='PlacementAPs.cprof')

    ## python ../PlacementAPs.py | egrep "(tottime)|(PlacementAPs.py)" | tee ../cProfile/PlacementAPs.py_COM-JIT.txt
    ## cat ../cProfile/PlacementAPs.py_COM-JIT.txt | sort -k 2 -r

    # python PlacementAPs.py | egrep '(ncalls)|(PlacementAPs)'
    # https://julien.danjou.info/blog/2015/guide-to-python-profiling-cprofile-concrete-case-carbonara

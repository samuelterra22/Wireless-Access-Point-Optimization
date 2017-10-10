import math
import tkinter as tk
from datetime import datetime
from math import sqrt, pi, log10, exp
from random import random

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import pygame

from numba import autojit, prange, cuda, jit
import numba

"""
Classe que realiza a simulação da propagação do sinal wireless de determinado ambiente 2D de acordo com um Access
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

    escala = 7

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
## @numba.jit("float64( int32[2], int32[2], List(List(int64)) )", target='parallel')
## @numba.jit(target='cpu', forceobj=True)
@jit
def absorption_in_walls(apX, apY, destinyX, destinyY):
    intersections = 0

    size = len(floor_plan)

    for i in range(size):
        # Coordenadas da parede

        if closed_segment_intersect(apX, apY, destinyX, destinyY, floor_plan[i][0], floor_plan[i][1], floor_plan[i][2],
                                    floor_plan[i][3]):
            intersections += 1

    intersecoes_com_paredes = intersections / 2

    miliWatts_absorvido_por_parede = 1

    return intersecoes_com_paredes * miliWatts_absorvido_por_parede


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
    return sqrt(pow((x1 - x2), 2.0) + pow((y1 - y2), 2.0))

@jit
def log_distance(d0, d, gamma):
    """
    Modelo logaritmo de perda baseado em resultados experimentais. Independe da frequência do sinal transmitido
    e do ganho das antenas transmissora e receptora.
    Livro Comunicações em Fio - Pricipios e Práticas - Rappaport (páginas 91-92).
    :param d0: Distância do ponto de referência d0.
    :param d: Distância que desejo calcular a perda do sinal.
    :param gamma: Valor da constante de propagação que difere para cada tipo de ambiente.
    :return: Retorna um float representando a perda do sinal entre a distância d0 e d.
    """
    # return path_loss(d) + 10 * gamma * log10(d / d0)
    return 17 - (60 + 10 * gamma * log10(d / d0))  # igual está na tabela

@jit
def tree_par_log(x):
    return -17.74321 - 15.11596 * math.log(x + 2.1642)


@jit
def propagation_model(x, y, apX, apY):
    d = calc_distance(x, y, apX, apY)

    loss_in_wall = 0

    loss_in_wall = absorption_in_walls(apX, apY, x, y)

    if d == 0:
        d = 1

    # value = log_distance(1, d, gamma)
    value = tree_par_log(d) - loss_in_wall
    # value = tree_par_log(d)

    return value

@jit
def objective_function(matrix):
    """
    Função objetivo para a avaliação da solução atual.
    :param matrix: Matriz a ser avaliada.
    :return: Retorna a soma de todos os elementos da metriz.
    """

    return abs(np.sum(matrix))


propagation_model_gpu = cuda.jit(device=True)(propagation_model)


@cuda.jit
def simulate_kernel(apX, apY, matrix_results):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param access_point: Access Point com a sua posição.
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, WIDTH, gridX):
        for y in range(startY, HEIGHT, gridY):
            matrix_results[x][y] = propagation_model_gpu(x, y, apX, apY)


def get_point_in_circle(pointX, pointY, ray, round_values=True, num=1, absolute_values=True):
    """
    Método por retorna um ponto ou conjunto de pontos dentro de um determinado raio de um ponto.
    :param point: Ponto contendo posição [x, y] de referência do ponto.
    :param ray: Valor do raio desejado.
    :param round_values: Flag que informa se o(s) ponto(s) serão arredondados. Geralmente será usando para retornar
    valores discretos para posições da matriz.
    :param absolute_values: Flag que informa se o(s) ponto(s) serão absolutos (positivos).
    :param num: Número de pontos que deseja gerar. Gera um ponto como default.
    :param debug: Flag que quando informada True, printa na tela o(s) ponto(s) gerados e a distância do ponto de
    referência.
    :return: Um ponto ou um conjunto de pontos do tipo float.
    """

    t = np.random.uniform(0.0, 2.0 * np.pi, num)
    r = ray * np.sqrt(np.random.uniform(0.0, 1.0, num))

    x = r * np.cos(t) + pointX
    y = r * np.sin(t) + pointY

    # Converte todos os valores negativos da lista em positivos
    if absolute_values:
        x = [abs(k) for k in x]
        y = [abs(k) for k in y]

    if round_values:
        x = [round(k) for k in x]
        y = [round(k) for k in y]

    # Verifica se o retorno será um ponto único ou uma lista de pontos.
    if num == 1:
        return [x[0], y[0]]
    else:
        return [x, y]

@jit
def perturb(SX, SY):
    """
     Função que realiza uma perturbação na Solução S.
     Solução pode ser perturbada em um raio 'r' dentro do espaço de simulação.
    :param S: Ponto atual.
    :return: Retorna um ponto dentro do raio informado.
    """
    # Obtem um ponto aleatorio em um raio de X metros
    return get_point_in_circle(SX, SY, 10)


@jit
def f(pointX, pointY):
    """
    Valor da função objetivo correspondente á configuração x;
    :param x: Ponto para realizar a simulação.
    :return: Retorna um numero float representando o valor da situação atual.
    """
    g_matrix = np.zeros(shape=(WIDTH, HEIGHT), dtype=np.float64)

    blockDim = (48, 8)
    gridDim = (32, 16)

    d_matrix = cuda.to_device(g_matrix)

    simulate_kernel[gridDim, blockDim](pointX, pointY, d_matrix)

    d_matrix.to_host()

    return objective_function(g_matrix)

def simulated_annealing(x0, y0, M, P, L, T0, alpha):
    """
    :param T0: Temperatura inicial.
    :param S0: Configuração Inicial (Entrada) -> Ponto?.
    :param M: Número máximo de iterações (Entrada).
    :param P: Número máximo de Perturbações por iteração (Entrada).
    :param L: Número máximo de sucessos por iteração (Entrada).
    :param alpha: Factor de redução da temperatura (Entrada).
    :return: Retorna um ponto sendo o mais indicado.
    """
    S = [x0, y0]
    T = T0
    j = 1

    fS = f(S[0], S[1])

    # Loop principal – Verifica se foram atendidas as condições de termino do algoritmo
    while True:
        i = 1
        nSucesso = 0

        # Loop Interno – Realização de perturbação em uma iteração
        while True:

            # Tera que mandar o ponto atual e a matriz (certeza?) tbm. Realiza a seleção do ponto.
            Si = perturb(S[0], S[1])
            fSi = f(Si[0], Si[1])

            # showSolution(Si)
            # print("[\t" + (str(round((100 - 100 * fSi / fS) * 100, 1))) + "\t] S: " + str(S) + "\t Si: " + str(Si))

            # Verificar se o retorno da função objetivo está correto. f(x) é a função objetivo
            deltaFi = fSi - fS

            # print("deltaFi: " + str(deltaFi))

            ## Minimização: deltaFi >= 0
            ## Maximização: deltaFi <= 0
            # Teste de aceitação de uma nova solução
            if (deltaFi <= 0) or (exp(-deltaFi / T) > random()):  # randomize()):
                # print("Ponto escolhido: " + str(Si))
                ## LEMBRETE: guardar o ponto anterior, S_prev = S (para ver o caminho do Si pro S_prev)
                S = Si
                fS = fSi
                nSucesso = nSucesso + 1

                # showSolution(S)
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
    # showSolution(S)
    # print("invocacoes de f(): " + str(contador_uso_func_objetivo))
    return S


########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################
if __name__ == '__main__':

    walls = read_walls_from_dxf("/home/samuel/PycharmProjects/TCC/DXFs/bloco-A-l.dxf")

    floor_plan = np.array(walls, dtype=np.float64)

    WIDTH = 500
    HEIGHT = 500

    comprimento_planta = 800
    largura_planta = 600
    precisao = 1  # metro

    escala = HEIGHT / largura_planta

    # tamanho da matriz = dimensão da planta / precisão

    proporcao_planta = comprimento_planta / largura_planta
    # WIDTH = int(HEIGHT * proporcao_planta)

    access_point = [0, 0]

    ## fixo, procurar uma fórmula para definir o max_iter em função do tamanho da matriz (W*H)
    max_inter = 600

    ## p
    max_pertub = 5

    ## v
    num_max_succ = 80

    ## a
    alpha = .85

    ## t
    temp_inicial = 300

    ## máximo de iterações do S.A.
    max_SA = 100

    print("\nIniciando simulação com simulated Annealing com a seguinte configuração:")
    print("Ponto inicial:\t\t\t\t\t" + str([access_point[0], access_point[1]]))
    print("Númeto máximo de iterações:\t\t\t" + str(max_inter))
    print("Número máximo de pertubações por iteração:\t" + str(max_pertub))
    print("Número máximo de sucessos por iteração:\t\t" + str(num_max_succ))
    print("Temperatura inicial:\t\t\t\t" + str(temp_inicial))
    print("Decaimento da teperatura com α=\t\t\t" + str(alpha))
    print("Repetições do Simulated Annealing:\t\t" + str(max_SA))
    input("Aperte qualquer tecla para iniciar.")

    bests = []

    for i in range(max_SA):
        print("Calculando o melhor ponto [" + str(i) + "]")
        bests.append(
            simulated_annealing(access_point[0], access_point[1], max_inter, max_pertub, num_max_succ, temp_inicial,
                                alpha))

    # Media das colunas, média dos melhores pontos
    best_mean = np.mean(bests, axis=0)

    # time_seconds = (fim - inicio).seconds
    # time_minutes = time_seconds / 60

    # print("\nInicio: \t" + str(inicio.time()))
    # print("Fim: \t\t" + str(fim.time()))
    # print("Duração: \t" + str(time_seconds) + " segundos (" + str(round(time_minutes, 2)) + " minutos).\n")

    print("Melhor ponto sugerido pelo algoritmo: " + str(best_mean))

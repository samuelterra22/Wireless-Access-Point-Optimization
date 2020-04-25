#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random as rd
import sys
from random import random

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import pygame
from colour import Color
from math import sqrt, log10, exp
from numba import cuda, jit

from utils.propagations_models import five_par_log_model

"""
Algoritmo que realiza a simulação da propagação do sinal wireless de determinado ambiente 2D de acordo com um Access
Point Informado.
"""


@jit
def read_walls_from_dxf(dxf_file_path, dxf_scale):
    """
    Método responsável por ler um arquivo DXF e filtrar pela camada ARQ as paredes do ambiente.
    :param dxf_scale:
    :param dxf_file_path: Caminho do arquivo de entrada, sendo ele no formato DFX.
    :return: Retorna uma lista contendo em cada posição, uma lista de quatro elementos, sendo os dois primeiros
    referêntes ao ponto inicial da parede e os dois ultimo referênte ao ponto final da parede.
    """
    dwg = ezdxf.readfile(dxf_file_path)

    dxf_walls = []

    model_space = dwg.modelspace()

    min_x = -1
    min_y = -1
    for e in model_space:
        if e.dxftype() == 'LINE' and e.dxf.layer == 'ARQ':
            if e.dxf.start[0] < min_x or min_x == -1:
                min_x = e.dxf.start[0]
            if e.dxf.start[1] < min_y or min_y == -1:
                min_y = e.dxf.start[1]

    for e in model_space:
        if e.dxftype() == 'LINE' and e.dxf.layer == 'ARQ':
            line = [
                int((e.dxf.start[0] - min_x) * dxf_scale),
                int((e.dxf.start[1] - min_y) * dxf_scale),
                int((e.dxf.end[0] - min_x) * dxf_scale),
                int((e.dxf.end[1] - min_y) * dxf_scale)
            ]
            dxf_walls.append(line)

    return dxf_walls


@jit
def side(a_x, a_y, b_x, b_y, c_x, c_y):
    """
    Returns a position of the point c relative to the line going through a and b
        Points a, b are expected to be different.
    :param c_y:
    :param c_x:
    :param b_y:
    :param b_x:
    :param a_y:
    :param a_x:
    :return:
    """
    d = (c_y - a_y) * (b_x - a_x) - (b_y - a_y) * (c_x - a_x)
    return 1 if d > 0 else (-1 if d < 0 else 0)


@jit
def is_point_in_closed_segment(a_x, a_y, b_x, b_y, c_x, c_y):
    """
    Returns True if c is inside closed segment, False otherwise.
        a, b, c are expected to be collinear
    :param c_y:
    :param c_x:
    :param b_y:
    :param b_x:
    :param a_y:
    :param a_x:
    :return: Retorna valor booleano True se for um ponto fechado por segmento de reta. Caso contrario retorna False.
    """
    if a_x < b_x:
        return a_x <= c_x <= b_x
    if b_x < a_x:
        return b_x <= c_x <= a_x

    if a_y < b_y:
        return a_y <= c_y <= b_y
    if b_y < a_y:
        return b_y <= c_y <= a_y

    return a_x == c_x and a_y == c_y


@jit
def closed_segment_intersect(a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y):
    """
    Verifies if closed segments a, b, c, d do intersect.
    :param a_x:
    :param a_y:
    :param b_x:
    :param b_y:
    :param c_x:
    :param c_y:
    :param d_x:
    :param d_y:
    :return:
    """
    if (a_x == b_x) and (a_y == b_y):
        return (a_x == c_x and a_y == c_y) or (a_x == d_x and a_y == d_y)
    if (c_x == d_x) and (c_y == d_y):
        return (c_x == a_x and c_y == a_y) or (c_x == b_x and c_y == b_y)

    s1 = side(a_x, a_y, b_x, b_y, c_x, c_y)
    s2 = side(a_x, a_y, b_x, b_y, d_x, d_y)

    # All points are collinear
    if s1 == 0 and s2 == 0:
        return \
            is_point_in_closed_segment(a_x, a_y, b_x, b_y, c_x, c_y) or is_point_in_closed_segment(a_x, a_y, b_x, b_y,
                                                                                                   d_x, d_y) or \
            is_point_in_closed_segment(c_x, c_y, d_x, d_y, a_x, a_y) or is_point_in_closed_segment(c_x, c_y, d_x, d_y,
                                                                                                   b_x, b_y)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    s1 = side(c_x, c_y, d_x, d_y, a_x, a_y)
    s2 = side(c_x, c_y, d_x, d_y, b_x, b_y)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    return True


@jit
def absorption_in_walls(ap_x, ap_y, destiny_x, destiny_y, floor_plan_model):
    intersections = 0

    size = len(floor_plan_model)

    for i in range(size):
        # Coordenadas da parede

        if closed_segment_intersect(ap_x, ap_y, destiny_x, destiny_y, floor_plan_model[i][0], floor_plan_model[i][1],
                                    floor_plan_model[i][2],
                                    floor_plan_model[i][3]):
            intersections += 1

    intersections_with_walls = intersections

    # parede de concredo, de 8 a 15 dB. Por conta da precision em casas decimais do float32, é melhor pegar a ordem de
    # magnitude com o dBm do que tentar usar o valor exato com mW
    # dbm_absorbed_by_wall = 8 ## AGORA É UMA CONSTANTE GLOBAL

    return intersections_with_walls * dbm_absorbed_by_wall


@jit
def mw_to_dbm(mw):
    """
    Método que converte a potência recebida dada em mW para dBm
    :param mw: Valor em miliwatts.
    :return: Valor de miliwatts convertido para decibéis.
    """
    return 10. * log10(mw)


@jit
def dbm_to_mw(dbm):
    """
    Método que converte a potência recebida dada em dBm para mW.
    :param dbm: Valor em decibéis.
    :return: Valor de decibéis convertidos em miliwatts.
    """
    return 10 ** (dbm / 10.)


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
    return sqrt(pow((x1 - x2), 2.0) + pow((y1 - y2), 2.0)) * precision


@jit
def propagation_model(x, y, ap_x, ap_y, floor_plan_model):
    d = calc_distance(x, y, ap_x, ap_y)

    loss_in_wall = absorption_in_walls(ap_x, ap_y, x, y, floor_plan_model)

    if d == 0:
        d = 1

    # CUIDADO: um modelo de propagação pessimista prende o SA se a FO não for ajustada

    # value = log_distance(d, 3, 11, -72, pt_dbm) - loss_in_wall
    # value = log_distance(d, 3,  1, -60, pt_dbm) - loss_in_wall
    # value = log_distance(d, 3, 10, -69, pt_dbm) - loss_in_wall
    value = five_par_log_model(pt_dbm, d) - loss_in_wall
    # value = four_par_log(pt_dbm, d) - loss_in_wall

    # TODO teste
    # return dbm_to_mw(value)

    return value


@jit
def objective_function(matrix):
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
    #         # if value < SENSITIVITY:
    #         #     g += -1
    #         # else:
    #         #     g += value
    #
    # return g
    # return abs(np.sum(np.power(10, matrix)))
    # return pow(10, x)

    # TODO: Penalizar os valores que estão abaixo da sensibilidade.
    # fo = abs(np.sum(matrix))

    # acima da sensibilidade
    fo = 0
    for line in matrix:
        for value in line:
            if value >= SENSITIVITY:
                fo += 1

    coverage_percent = (fo / TOTAL_OF_POINTS) * 100  # porcentagem de cobertura
    shadow_percent = 100 - coverage_percent  # porcentagem de sombra

    # return coverage_percent 					 ## maximiza a cobertura
    # return (-1 * pow(shadow_percent,2))			 ## miminiza as sombras, penalizadas
    # return pow(coverage_percent,2)				 ## maximiza a cobertura, difereciando mais os bons resultados

    # return ( 2*coverage_percent - shadow_percent )

    fo_alpha = 7
    return fo_alpha * coverage_percent - (10 - fo_alpha) * shadow_percent  # pesos 7 pra 3
    # return (0.7 * coverage_percent - 0.3 * shadow_percent)  # pesos 7 pra 3

    # TODO testing VALADAO
    # return abs(np.sum(matrix))

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
    matrix_width = len(matrix)
    matrix_height = len(matrix[0])

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, matrix_width, grid_x):
        for y in range(start_y, matrix_height, grid_y):
            soma += matrix[x][y]

            #         if matrix[x][y] >= SENSITIVITY:
            #             soma += 1

            # soma = ((soma / TOTAL_OF_POINTS) * 100)


@cuda.jit
def simulate_kernel(ap_x, ap_y, matrix_results, floor_plan_model):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param floor_plan_model:
    :param ap_y:
    :param ap_x:
    :param matrix_results:
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, WIDTH, grid_x):
        for y in range(start_y, HEIGHT, grid_y):
            matrix_results[x][y] = propagation_model_gpu(x, y, ap_x, ap_y, floor_plan_model)


propagation_model_gpu = cuda.jit(device=True)(propagation_model)


@jit
def simulate_cpu(ap_x, ap_y, matrix_results, floor_plan_model):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param floor_plan_model:
    :param matrix_results:
    :param ap_y:
    :param ap_x:
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    for x in range(WIDTH):
        for y in range(HEIGHT):
            matrix_results[x][y] = propagation_model(x, y, ap_x, ap_y, floor_plan_model)

    return matrix_results


@jit
def get_point_in_circle(point_x, point_y, ray):
    """
    Método reponsavel por retornar um ponto ou conjunto de pontos dentro de um determinado raio de um ponto.
    :param point_y:
    :param point_x:
    :param ray: Valor do raio desejado.
    valores discretos para posiÃ§Ãµes da matriz.
    :return: Um ponto ou um conjunto de pontos do tipo float.
    """
    num = 1

    t = np.random.uniform(0.0, 2.0 * np.pi, num)
    r = ray * np.sqrt(np.random.uniform(0.0, 1.0, num))

    x = r * np.cos(t) + point_x
    y = r * np.sin(t) + point_y

    # Converte todos os valores negativos da lista em positivos

    x = round(abs(x[0]))
    y = round(abs(y[0]))

    # Verifica se o valor estrapolou as dimensões da simulação
    if x > WIDTH:
        x = WIDTH

    if y > HEIGHT:
        y = HEIGHT

    return list([x, y])


@jit
def disturb_array(s_array, size):
    """
     Função que realiza uma perturbação na Solução S.
     Solução pode ser perturbada em um raio 'r' dentro do espaço de simulação.
    :param size:
    :param s_array:
    :return: Retorna um ponto dentro do raio informado.
    """
    new_s = np.empty([num_aps, 2], np.float32)

    for i in range(size):
        # Obtem um ponto aleatorio em um raio de X metros
        new_s[i] = get_point_in_circle(s_array[i][0], s_array[i][1], DISTURBANCE_RADIUS)

    return new_s


@jit
def disturb_solution(solution):
    """
     Função que realiza uma perturbação na Solução S.
     Solução pode ser perturbada em um raio 'r' dentro do espaço de simulação.
    :param solution: Ponto atual.
    :return: Retorna um ponto dentro do raio informado.
    """

    return get_point_in_circle(solution[0], solution[1], DISTURBANCE_RADIUS)


@jit
def evaluate_array(s_array, size):
    propagation_matrices = []
    for i in range(size):
        propagation_matrices.append(simulates_propagation(s_array[i][0], s_array[i][1]))

    # simplesmente guloso VALADAO testing
    overlaid_matrix = solution_overlap_max(propagation_matrices, size)

    # penaliza APs muito proximos (CUIDADO: junto com FO % de cobertura prender o SA)
    # overlaid_matrix = solution_overlap_div_dbm(propagation_matrices, size)

    return objective_function(overlaid_matrix), propagation_matrices


@jit
def solution_overlap_max(propagation_array, size):
    max_value = propagation_array[0]
    for i in range(1, size):
        max_value = np.maximum(propagation_array[i], max_value)

    return max_value


@jit
def solution_overlap_sub(propagation_array, size):
    sub = propagation_array[0]
    for i in range(1, size):
        sub = np.subtract(propagation_array[i], sub)

    return sub


@jit
def solution_overlap_div_dbm(propagation_array, size):
    # verificar se é veridico
    if size == 1:
        return propagation_array[0]

    matrix_min_x = propagation_array[0]
    matrix_max = propagation_array[0]

    for i in range(1, size):
        matrix_min_x = np.minimum(propagation_array[i], matrix_min_x)
        matrix_max = np.maximum(propagation_array[i], matrix_max)

    # pois ao subtrair dbm, deve ser o maior/menor
    sub = np.divide(matrix_max, matrix_min_x)

    return sub


@jit
def simulates_propagation_cpu(ap_x, ap_y):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param ap_y:
    :param ap_x:
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    matrix_results = np.empty([WIDTH, HEIGHT], np.float32)

    return simulate_cpu(ap_x, ap_y, matrix_results, floor_plan)


@jit
def simulates_propagation_gpu(point_x, point_y):
    """
    Valor da função objetivo correspondente á configuração x;
    :param point_x:
    :param point_y: Ponto para realizar a simulação.
    :return: Retorna um numero float representando o valor da situação atual.
    """
    g_matrix = np.zeros(shape=(WIDTH, HEIGHT), dtype=np.float32)

    block_dim = (48, 8)
    grid_dim = (32, 16)

    d_matrix = cuda.to_device(g_matrix)

    simulate_kernel[grid_dim, block_dim](point_x, point_y, d_matrix, floor_plan)

    d_matrix.to_host()

    return g_matrix


@jit
def simulates_propagation(point_x, point_y):
    """
    Método resposável por realizar a simulação da propagação de acordo com o ambiente escolhido (CPU ou GPU)
    :param point_x:
    :param point_y:
    :return:
    """

    if ENVIRONMENT == "GPU":
        # with GPU CUDA Threads
        return simulates_propagation_gpu(point_x, point_y)

    elif ENVIRONMENT == "CPU":
        #  with CPU Threads
        return simulates_propagation_cpu(point_x, point_y)
    else:
        # exit(-1)
        sys.exit("(ERROR) Nenhum ambiente de execução bem definido.")


@jit
def objective_function_mw(array_matrix):
    matrix = solution_overlap_max(array_matrix, len(array_matrix))

    sum_matrix = 0

    for line in matrix:
        for value in line:
            sum_matrix += dbm_to_mw(value)

    return sum_matrix


def simulated_annealing(size, M, P, L, T0, alpha):
    """
    :param size:
    :param T0: Temperatura inicial.
    :param M: Número máximo de iterações.
    :param P: Número máximo de Perturbações por iteração.
    :param L: Número máximo de sucessos por iteração.
    :param alpha: Factor de redução da temperatura.
    :return: Retorna um ponto sendo o mais indicado.
    """

    # cria Soluções iniciais com pontos aleatórios para os APs
    s_array = np.empty([size, 2], np.float32)

    for i in range(size):  # VALADAO testing
        if INITIAL_POSITION == RANDOM:
            s_array[i] = [rd.randrange(0, WIDTH), rd.randrange(0, HEIGHT)]
        elif INITIAL_POSITION == CENTER:
            s_array[i] = [WIDTH * 0.5, HEIGHT * 0.5]

    s0 = s_array.copy()
    print("Solução inicial:\n" + str(s0))

    result = evaluate_array(s_array, size)
    f_s = result[0]

    T = T0
    j = 1

    i_ap = 0

    # Armazena a MELHOR solução encontrada
    best_s_array = s_array.copy()
    best_fs = f_s
    # BEST_matrix_FO = result[1]

    # Loop principal – Verifica se foram atendidas as condições de termino do algoritmo
    while True:
        i = 1
        n_success = 0

        # Loop Interno – Realização de perturbação em uma iteração
        while True:

            initial_solutions_array = s_array.copy()

            # a cada iteração do SA, disturb_solution um dos APs
            i_ap = (i_ap + 1) % num_aps

            initial_solutions_array[i_ap] = disturb_solution(s_array[i_ap])

            # retorna a FO e suas matrizes
            result = evaluate_array(initial_solutions_array, num_aps)
            f_si = result[0]
            # matrix_FO = result[1]

            # Cuidado pois fica demasiado lento o desempenho do SA
            # if ANIMATION_STEP_BY_STEP:
            #   show_solution(s_array, py_game_display_surf)

            # Verificar se o retorno da função objetivo está correto. f(x) é a função objetivo
            delta_fi = f_si - f_s

            # Minimização: delta_fi >= 0
            # Maximização: delta_fi <= 0
            # Teste de aceitação de uma nova solução
            if (delta_fi <= 0) or (exp(-delta_fi / T) > random()):

                s_array = initial_solutions_array.copy()
                f_s = f_si

                n_success = n_success + 1

                # Cuidado pois fica demasiado lento o desempenho do SA
                # if ANIMATION_BEST_PLACES:
                #   show_solution(s_array, py_game_display_surf)

                if f_s > best_fs:
                    best_fs = f_s
                    best_s_array = s_array.copy()

                    if ANIMATION_BESTS:
                        show_solution(s_array, py_game_display_surf)

                FOs.append(f_s)

            i = i + 1

            if (n_success >= L) or (i > P):
                break

        # Atualização da temperatura (Deicaimento geométrico)
        T = alpha * T

        # Atualização do contador de iterações
        j = j + 1

        if (n_success == 0) or (j > M):
            break

    print("Distância da solução inicial:\t\t\t\t\t" + str(solution_overlap_sub(s_array, num_aps)))

    print("FO last cand:   " + '{:.3e}'.format(float(f_si)))
    print("FO local best:  " + '{:.3e}'.format(float(f_s)))
    print("FO global best: " + '{:.3e}'.format(float(best_fs)))

    # FOs.append( objective_function_mw(BEST_matrix_FO) )
    # FOs.append( mw_to_dbm(objective_function_mw(BEST_matrix_FO)) )
    FOs.append(best_fs)

    return best_s_array


def hex_to_rgb(hex_value):
    """
    Método responsável por converter uma cor no formato hexadecial para um RGB.
    :param hex_value: Valor em hexadecimal da cor.
    :return: Tupla representando a cor em formato RGB.
    """
    hex_value = str(hex_value).lstrip('#')
    lv = len(hex_value)
    return tuple(int(hex_value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def draw_line(py_game_display_surf_value, x1, y1, x2, y2, color):
    """
    Método responsável por desenhar uma linha reta usando o PyGame de acordo com a posição de dois pontos.
    :param py_game_display_surf_value:
    :param x1: Valor de X no ponto 1.
    :param y1: Valor de Y no ponto 1.
    :param x2: Valor de X no ponto 2.
    :param y2: Valor de Y no ponto 2.
    :param color: Cor que a linha irá ter.
    :return: None
    """
    pygame.draw.line(py_game_display_surf_value, color, (x1, y1), (x2, y2))


def print_py_game(matrix_results, access_points, py_game_display_surf_value):
    """
    Método responsável por desenhar a simulação usando o PyGame.
    :param py_game_display_surf_value:
    :param access_points:
    :param matrix_results: Matrix float contendo os resultados da simulação.
    :return: None.
    """

    matrix_max_value = matrix_results.max()

    # matrix_max_value = -30
    matrix_min_value = -100

    # Lê os valores da matriz que contêm valores calculados e colore
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = get_color_of_interval(matrix_results[x][y], matrix_max_value, matrix_min_value)
            draw_point(py_game_display_surf_value, color, x, y)

    # Printa de vermelho a posição dos Access Points
    for ap in access_points:
        draw_point(py_game_display_surf_value, RED, ap[0], ap[1])

    # draw_floor_plan(floor_plan)

    # Atualiza a janela do PyGame para que exiba a imagem
    pygame.display.update()


def draw_point(py_game_display_surf_value, color, x, y):
    """
    Método responsável por desenhar um ponto usando o PyGame de acordo com a posição (x,y).
    :param py_game_display_surf_value:
    :param color: A cor que irá ser o ponto.
    :param x: Posição do ponto no eixo X.
    :param y: Posição do ponto no eixo Y.
    :return: None.
    """
    pygame.draw.line(py_game_display_surf_value, color, (x, y), (x, y))


def size_of_floor_plan(floor_plan_model):
    """
    Método responsável por obter as dimenções da planta
    :param floor_plan_model:
    :return:
    """
    x_max = y_max = 0

    for lines in floor_plan_model:
        if lines[0] > x_max:
            x_max = lines[0]
        if lines[2] > x_max:
            x_max = lines[2]

        if lines[1] > y_max:
            y_max = lines[1]
        if lines[3] > y_max:
            y_max = lines[3]

    return [x_max, y_max]


def draw_floor_plan(floor_plan_model, py_game_display_surf_value):
    for line in floor_plan_model:
        draw_line(py_game_display_surf_value, line[0], line[1], line[2], line[3], WHITE)

    # Atualiza a janela do PyGame para que exiba a imagem
    pygame.display.update()


def get_percentage_of_range(min_value, max_value, x):
    """
    Método responsável por retornar a porcentagem de acordo com um respectivo intervalo.
    :param min_value: Valor mínimo do intervalo.
    :param max_value: Valor máximo do intervalo.
    :param x: Valor que está no intervalo de min-max que deseja saber sua respectiva porcentagem.
    :return: Retorna uma porcentagem que está de acordo com o intervalo min-max.
    """

    return ((x - min_value) / (max_value - min_value)) * 100


def get_value_in_list(percent, list_numbers):
    """
    Método retorna o valor de uma posição de uma lista. A posição é calculada de acordo a porcentagem.
    :param percent: Valor float representando a porcentagem.
    :param list_numbers: Lista com n números.
    :return: Retorna a cor da posição calculada.
    """
    position = (percent / 100) * len(list_numbers)
    if position < 1:
        position = 1
    elif position >= len(list_numbers):
        position = len(list_numbers)
    return hex_to_rgb(list_numbers[int(position - 1)])


def get_color_of_interval(x, max_value=-30, min_value=-100):
    """
    Este método retorna uma cor de acordo com o valor que está entre o intervalo min-max. Em outras palavras,
    este método transforma um número em uma cor dentro de uma faixa informada.
    :param min_value: Valor mínimo do intervalo.
    :param max_value: Valor máximo do intervalo.
    :param x: Valor que está dentro do intervalo e que deseja saber sua cor.
    :return: Retorna uma tupla representando um cor no formato RGB.
    """

    if PAINT_BLACK_BELOW_SENSITIVITY and x < SENSITIVITY:
        return BLACK

    percentage = get_percentage_of_range(min_value, max_value, x)
    color = get_value_in_list(percentage, COLORS)

    return color


def show_solution(s_array, py_game_display_surf_value):
    propagation_matrices = []

    for i in range(len(s_array)):
        propagation_matrices.append(simulates_propagation(s_array[i][0], s_array[i][1]))

    # propagation = solution_overlap_ADD(propagation_matrices, len(s_array))
    propagation = solution_overlap_max(propagation_matrices, len(s_array))

    # generate_summary(s_array)

    print_py_game(propagation, s_array, py_game_display_surf_value)

    draw_floor_plan(walls, py_game_display_surf_value)

    pygame.display.update()


def get_color_gradient(steps=250):
    cores = list(Color("red").range_to(Color("green"), steps))
    cores.pop(0)
    cores.pop(len(cores) - 1)
    return cores


def show_configs():
    print("\nOtimizacao via Simulated Annealing com a seguinte configuracao:" + "\n")
    print("\tNumero maximo de iteracoes:\t\t\t" + str(max_inter))
    print("\tNumero maximo de pertubacoes por iteracao:\t" + str(max_disturbances))
    print("\tNumero maximo de sucessos por iteracao:\t\t" + str(num_max_success))
    print("\tTemperatura inicial:\t\t\t\t" + str(initial_temperature))
    print("\tDecaimento da teperatura com α=\t\t\t" + str(alpha))
    print("\tRaio de perturbacao:\t\t\t\t" + str(int(DISTURBANCE_RADIUS)))

    print("\tExecucoes do otimziador: \t\t\t" + str(max_SA))
    print("\nHardware de simulacao:\t" + str(ENVIRONMENT) + "\n")

    print("\nSimulacao do ambiente com a seguinte configuracao:" + "\n")
    print("\tSimulando ambiente com:  \t\t" + str(WIDTH) + " x " + str(HEIGHT) + " pixels")
    print("\tEscala de simulacao:     \t\t1 px : " + '{:.4f}'.format(float((1 / scale))) + " metros")

    print("\tQuantidade de APs:       \t\t" + str(num_aps))
    print("\tPotencia de cada APs:    \t\t" + str(pt_dbm) + " dBm")

    if INITIAL_POSITION == RANDOM:
        print("\tPosicao inicial dos APs: \t\tALEATORIA")
    elif INITIAL_POSITION == CENTER:
        print("\tPosicao inicial dos APs: \t\tCENTRALIZADA (W/2, H/2)")
    elif INITIAL_POSITION == CUSTOM:
        print("\tPosicao inicial dos APs: \t\tCUSTOMIZADA")


def run():
    best_solution = simulated_annealing(num_aps, max_inter, max_disturbances, num_max_success, initial_temperature,
                                        alpha)
    evaluate_array(best_solution, len(best_solution))

    # Gera resumo da simulação
    generate_summary(best_solution)

    print("\nDesenhando resultado da simulação...")
    if ANIMATION_STEP_BY_STEP or ANIMATION_BEST_PLACES or ANIMATION_BESTS:
        show_solution(best_solution, py_game_display_surf)


def fixed_aps(best_solution):
    evaluate_array(best_solution, len(best_solution))

    # Gera resumo da simulação
    generate_summary(best_solution)

    print("\nDesenhando resultado da simulação...")
    if ANIMATION_STEP_BY_STEP or ANIMATION_BEST_PLACES or ANIMATION_BESTS:
        show_solution(best_solution, py_game_display_surf)


def test_propagation():
    """
    Método usado apenas para fim de testes com a simulação em pontos específicos.
    :return: None.
    """
    test_ap_in_the_middle = [[int(WIDTH / 2), int(HEIGHT / 2)]]

    #
    if ANIMATION_STEP_BY_STEP or ANIMATION_BEST_PLACES or ANIMATION_BESTS:
        # Initialize PyGame
        pygame.init()

        # Set window size
        py_game_display_surf_value = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        show_solution(test_ap_in_the_middle, py_game_display_surf_value)


def generate_summary(s_array):
    length = len(s_array)

    print("\n****** Gerando sumarios dos resultados da simulacao ******")
    print("Numero de APs:\t" + str(length))
    print("Solução: ", s_array)

    propagation_matrices = []
    for i in range(length):
        propagation_matrices.append(simulates_propagation(s_array[i][0], s_array[i][1]))

    matrix = solution_overlap_max(propagation_matrices, length)

    above_sensitivity = [value for line in matrix for value in line if value >= SENSITIVITY]
    under_sensitivity = [value for line in matrix for value in line if value < SENSITIVITY]

    total = WIDTH * HEIGHT

    percent_cover_above_sensitivity = (len(above_sensitivity) / total) * 100
    percent_cover_under_sensitivity = (len(under_sensitivity) / total) * 100

    print("\nCOBERTURA DE SINAL WI-FI:")
    print("\t" + '{:.2f}'.format(float(percent_cover_above_sensitivity)) + "%\t com boa cobertura (sinal forte)")
    print("\t" + '{:.2f}'.format(
        float(percent_cover_under_sensitivity)) + "%\t de zonas de sombra (abaixo da sensibilidade)")

    range_1 = range_2 = range_3 = faixa4 = 0

    for line in matrix:
        for value in line:
            if value >= -67:  # ótimo
                range_1 += 1

            elif -67 > value >= -77:  # bom
                range_2 += 1

            elif -77 > value >= SENSITIVITY:  # ruim
                range_3 += 1

            elif value < SENSITIVITY:  # sem conectividade (zona de sombra)
                faixa4 += 1

    total = range_1 + range_2 + range_3 + faixa4  # deveria ser igual a WIDTH * HEIGHT

    percent_range_1 = range_1 / total * 100
    percent_range_2 = range_2 / total * 100
    percent_range_3 = range_3 / total * 100

    print("\n\tCobertura por FAIXAS de intensidade de sinal")
    print("\t\tsinal Otimo  \t" + '{:.1f}'.format(float(percent_range_1)) + "%")
    print("\t\tsinal Bom    \t" + '{:.1f}'.format(float(percent_range_2)) + "%")
    print("\t\tsinal Ruim   \t" + '{:.1f}'.format(float(percent_range_3)) + "%")

    if FOs:
        # Plota gráfico da função objetivo
        print("\n... gerando grafico do comportamento da FO.")
        plt.plot(FOs)
        plt.title("Comportamento do Simulated Annealing")
        plt.ylabel('Valor da FO')
        plt.xlabel('Solucao candidata')
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

    # OBS.: por conta da precision de casas decimais do float
    #        é melhor pegar a ordem de magnitude com o dBm do
    #        que tentar usar o valor exato com mW

    # Sensibilidade dos equipamentos receptores
    SENSITIVITY = -90

    # Gradiente de cores da visualização gráfica
    COLORS = get_color_gradient(16)  # 64, 32, 24, 16, 8

    PAINT_BLACK_BELOW_SENSITIVITY = True
    # PAINT_BLACK_BELOW_SENSITIVITY = False

    DBM_MIN_VALUE = np.finfo(np.float32).min

    # parede de concredo, de 8 a 15 dB.
    dbm_absorbed_by_wall = 8

    # Potência de transmissão de cada AP
    # pt_dbm = -14
    # pt_dbm = -17
    # pt_dbm = -20
    pt_dbm = -25
    # pt_dbm = -30

    # Quantidade de APs
    num_aps = 3

    # Constantes para controle da estratégia de posição inicial dos APs
    RANDOM = 0
    CENTER = 1
    CUSTOM = 3
    INITIAL_POSITION = CENTER

    ##################################################
    #  CONFIGURAÇÕES DO AMBIENTE E PLANTA-BAIXA

    LENGTH_BLOCK_A = 48.0
    LENGTH_BLOCK_B = 36.0
    LENGTH_BLOCK_C = 51.0

    LENGTH_BUILDING = LENGTH_BLOCK_A
    # LARGURA_EDIFICIO = ???

    # dxf_path = "./DXFs/bloco_a/bloco_A_planta baixa_piso1.dxf"
    dxf_path = "./DXFs/bloco_a/bloco_A_planta baixa_piso1_porta.dxf"

    # dxf_path = "./DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso1.dxf"
    # dxf_path = "./DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso2.dxf"
    # dxf_path = "./DXFs/bloco_c/com_porta/bloco_C_planta baixa_piso3.dxf"
    # dxf_path = "./DXFs/bloco_c/sem_porta/bloco_C_planta_baixa_piso1.dxf"
    # dxf_path = "./DXFs/bloco_c/sem_porta/bloco_C_planta baixa_piso2.dxf"
    # dxf_path = "./DXFs/bloco_c/sem_porta/bloco_C_planta baixa_piso3.dxf"

    # carrega para saber o comprimento da planta
    walls = read_walls_from_dxf(dxf_path, 1)
    floor_plan = np.array(walls, dtype=np.float32)

    floor_size = size_of_floor_plan(walls)
    floor_plan_length = floor_size[0]
    floor_plan_width = floor_size[1]

    ##################################################
    #  CONFIGURAÇÕES DO AMBIENTE SIMULADO

    ENVIRONMENT = "GPU"
    # ENVIRONMENT = "CPU"

    # Tamanho da simulação
    # SIMULATION_SIZE = 400
    SIMULATION_SIZE = 600

    # Ativa / Desativa a animação passo a passo da otimização
    # ANIMATION_STEP_BY_STEP   = True
    ANIMATION_STEP_BY_STEP = False

    # ANIMATION_BEST_PLACES = True
    ANIMATION_BEST_PLACES = False

    ANIMATION_BESTS = True
    # ANIMATION_BESTS = False

    ##################################################

    # Lista para guardar as funções objetivos calculadas durante a simulação
    FOs = []

    WIDTH = SIMULATION_SIZE
    HEIGHT = int(WIDTH * (floor_plan_width / floor_plan_length))
    scale = WIDTH / floor_plan_length
    precision = LENGTH_BUILDING / WIDTH

    TOTAL_OF_POINTS = WIDTH * HEIGHT

    # HEIGHT = SIMULATION_SIZE
    # WIDTH = int(HEIGHT * (floor_plan_length / floor_plan_width))
    # scale = HEIGHT / floor_plan_width
    # precision = LARGURA_EDIFICIO / WIDTH

    # RE-carrega utilizando a escala apropriada
    walls = read_walls_from_dxf(dxf_path, scale)
    floor_plan = np.array(walls, dtype=np.float32)
    ##################################################

    ##################################################
    #  CONFIGURAÇÕES DO OTIMIZADOR

    # fixo, procurar uma fórmula para definir o max_iter em função do tamanho da matriz (W*H)
    max_inter = 600
    # max_inter = 600 * (1 + num_aps)
    # max_inter = 600 * (10 * num_aps)
    # max_inter = TOTAL_OF_POINTS * 0.2

    # p - Máximo de perturbações
    max_disturbances = 5

    # DISTURBANCE_RADIUS = WIDTH * 0.0100
    # DISTURBANCE_RADIUS = WIDTH * 0.0175
    # DISTURBANCE_RADIUS = WIDTH * 0.0250
    # DISTURBANCE_RADIUS = WIDTH * 0.1100
    beta = 1
    DISTURBANCE_RADIUS = (1 / precision) * (beta + num_aps)  # VALADAO testing
    # DISTURBANCE_RADIUS = (1 / precision) * (1 + num_aps)  # VALADAO testing

    # v - Máximo de vizinhos
    # num_max_success = 80
    # num_max_success = 80 * 10
    # num_max_success = 80 * (beta + num_aps) * 3
    num_max_success = 240 * (beta + num_aps)

    # a - Alpha
    alpha = .85
    # alpha = .95

    # t - Temperatura
    initial_temperature = 300 * (beta + num_aps)
    # initial_temperature = 300 * (1 + num_aps) * 10

    # Máximo de iterações do S.A.
    max_SA = 1
    ##################################################

    # Visualização dos dados
    # Inicia o PyGame e configura o tamanho da janela

    # ANIMATION_STEP_BY_STEP   = True

    # Só inicializa a janela do PyGame se alguma flag estiver ativa
    if ANIMATION_STEP_BY_STEP or ANIMATION_BEST_PLACES or ANIMATION_BESTS:
        pygame.init()
        icon = pygame.image.load('images/icon.png')
        pygame.display.set_icon(icon)
        pygame.display.set_caption("Resultado Simulação - IFMG Campus Formiga")
        py_game_display_surf = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

    show_configs()
    # test_propagation()
    run()
    # fixed_aps([[1., 210.], [300., 225.], [385., 225.], [540, 225]])
    #
    # best_solution = [
    #     [WIDTH * 0.5, HEIGHT * 0.5]
    # ]
    #
    # show_solution(best_solution, py_game_display_surf)

    # profile.runctx('run()', globals(), locals(),'tese')
    # cProfile.run(statement='run()', filename='PlacementAPs.cprof')

    # python ../main.py | egrep "(tottime)|(main.py)" | tee ../cProfile/PlacementAPs.py_COM-JIT.txt
    # cat ../cProfile/PlacementAPs.py_COM-JIT.txt | sort -k 2 -r

    # python main.py | egrep '(ncalls)|(PlacementAPs)'
    # https://julien.danjou.info/blog/2015/guide-to-python-profiling-cprofile-concrete-case-carbonara

    # generate_summary([[50, 50]])

    input('\nAperte ESC para fechar a simulação.')

    # profile.runctx('run()', globals(), locals(),'tese')
    # cProfile.run(statement='run()', filename='PlacementAPs.cprof')

    # python ../main.py | egrep "(tottime)|(main.py)" | tee ../cProfile/PlacementAPs.py_COM-JIT.txt
    # cat ../cProfile/PlacementAPs.py_COM-JIT.txt | sort -k 2 -r

    # python main.py | egrep '(ncalls)|(PlacementAPs)'
    # https://julien.danjou.info/blog/2015/guide-to-python-profiling-cprofile-concrete-case-carbonara

import math
import tkinter as tk
from datetime import datetime
from math import sqrt, pi, log10, exp
from random import random

import ezdxf
import numpy as np
import pygame

from numba import cuda
from numba import *


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
def side(a, b, c):
    """
    Returns a position of the point c relative to the line going through a and b
        Points a, b are expected to be different.
    :param a: Ponto A.
    :param b: Ponto B.
    :param c: Ponto C.
    :return:
    """
    d = (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    return 1 if d > 0 else (-1 if d < 0 else 0)


@jit
def is_point_in_closed_segment(a, b, c):
    """
    Returns True if c is inside closed segment, False otherwise.
        a, b, c are expected to be collinear
    :param a: Ponto A.
    :param b: Ponto B.
    :param c: Ponto C.
    :return: Retorna valor booleano True se for um ponto fechado por segmento de reta. Caso contrario retorna False.
    """
    if a[0] < b[0]:
        return a[0] <= c[0] <= b[0]
    if b[0] < a[0]:
        return b[0] <= c[0] <= a[0]

    if a[1] < b[1]:
        return a[1] <= c[1] <= b[1]
    if b[1] < a[1]:
        return b[1] <= c[1] <= a[1]

    return a[0] == c[0] and a[1] == c[1]


@jit
def closed_segment_intersect(a, b, c, d):
    """ Verifies if closed segments a, b, c, d do intersect.
    """
    if a == b:
        return a == c or a == d
    if c == d:
        return c == a or c == b

    # TODO ao inves de invocar a funcao side, colocar a formula aqui
    s1 = side(a, b, c)
    s2 = side(a, b, d)

    # All points are collinear
    if s1 == 0 and s2 == 0:
        # TODO ao inves de invocar a funcao is_point_in_closed_segment, colocar a formula aqui
        return \
            is_point_in_closed_segment(a, b, c) or is_point_in_closed_segment(a, b, d) or \
            is_point_in_closed_segment(c, d, a) or is_point_in_closed_segment(c, d, b)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    s1 = side(c, d, a)
    s2 = side(c, d, b)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    return True


def absorption_in_walls(access_point, destiny, walls):
    intersections = 0

    for line in walls:
        # Coordenadas da parede
        wall_xy_a = line[0:2]
        wall_xy_b = line[2:4]

        if closed_segment_intersect(access_point, destiny, wall_xy_a, wall_xy_b):
            intersections += 1

    intersecoes_com_paredes = intersections / 2

    miliWatts_absorvido_por_parede = 1

    return intersecoes_com_paredes * miliWatts_absorvido_por_parede


def get_monitor_size():
    """
    Método que identifica o tamanho da tela do computador.
    :return: Retorna os valores de largura e altura.
    """
    root = tk.Tk()
    return root.winfo_screenwidth(), root.winfo_screenheight()


def mw_to_dbm(mW):
    """
    Método que converte a potência recebida dada em mW para dBm
    :param mW: Valor em miliwatts.
    :return: Valor de miliwatts convertido para decibéis.
    """
    return 10. * log10(mW)


def dbm_to_mw(dBm):
    """
    Método que converte a potência recebida dada em dBm para mW.
    :param dBm: Valor em decibéis.
    :return: Valor de decibéis convertidos em miliwatts.
    """
    return 10 ** (dBm / 10.)


def hex_to_rgb(hex):
    """
    Método responsável por converter uma cor no formato hexadecial para um RGB.
    :param hex: Valor em hexadecimal da cor.
    :return: Tupla representando a cor em formato RGB.
    """
    hex = str(hex).lstrip('#')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def draw_line(x1, y1, x2, y2, color):
    """
    Método responsável por desenhar uma linha reta usando o PyGame de acordo com a posição de dois pontos.
    :param x1: Valor de X no ponto 1.
    :param y1: Valor de Y no ponto 1.
    :param x2: Valor de X no ponto 2.
    :param y2: Valor de Y no ponto 2.
    :param color: Cor que a linha irá ter.
    :return: None
    """
    # pygame.draw.line(DISPLAYSURF, color, (x1, y1), (x2, y2))


def draw_point(color, x, y):
    """
    Método responsável por desenhar um ponto usando o PyGame de acordo com a posição (x,y).
    :param color: A cor que irá ser o ponto.
    :param x: Posição do ponto no eixo X.
    :param y: Posição do ponto no eixo Y.
    :return: None.
    """
    # pygame.draw.line(DISPLAYSURF, color, (x, y), (x, y))


def draw_floor_plan(floor_plan):
    for line in floor_plan:
        draw_line(line[0], line[1], line[2], line[3], WHITE)

    # Atualiza a janela do PyGame para que exiba a imagem
    pygame.display.update()


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
    return np.sqrt(np.power((x1 - x2), 2.0) + np.power((y1 - y2), 2.0))


def frequency():
    """
    Método responsável por calcular a frequência de acordo com o canal.
    :return: Frequência do canal.
    """
    return (2.407 + (5 * CHANNEL) / 1000) * 10 ** 9


def wave_length():
    """
    Método responsável por calcular o comprimento de onda como razão a velocidade da luz da frequência do canal.
    :return: Comprimento de onda de acordo com a frequência.
    """
    C = 299792458
    return C / frequency()


def path_loss(d):
    """
    Perda no caminho (Path Loss) mensurado em dB.
    :param d: Distâcia.
    :return: Perda no caminho.
    """
    return 20 * np.log10((4 * np.pi * d) / wave_length())


def two_ray_ground_reflection_model(Pt, Gt, Gr, Ht, Hr, d, L):
    """
    Pr
    """
    return (Pt * Gt * Gr * np.power(Ht, 2) * np.power(Hr, 2)) / (np.power(d, 4) * L)


def free_space_model(Pt, Gt, Gr, lamb, d, L):
    """
    Pr
    """
    return (Pt * Gt * Gr * (np.power(lamb, 2))) / (np.power((4 * pi), 2) * np.power(d, 2) * L)


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
    return 17 - (60 + 10 * gamma * np.log10(d / d0))  # igual está na tabela


@jit
def tree_par_log(x):
    return -17.74321 - 15.11596 * np.log(x + 2.1642)


@jit
def propagation_model(x, y, access_point):
    d = calc_distance(x, y, access_point[0], access_point[1])

    loss_in_wall = 0

    #loss_in_wall = absorption_in_walls(access_point=access_point, destiny=[x, y], walls=floor_plan)
    if d == 0:
        d = 1
    gamma = 5

    # value = log_distance(1, d, gamma)
    value = tree_par_log(d) - loss_in_wall
    # value = tree_par_log(d)

    return value


def print_matriz(matrix):
    """
    Método responsável por imprimir a matriz em um arquivo.
    :param matrix: Matriz N x M.
    :return: None.
    """
    print("Escrevendo matrix no arquivo de saida...")
    print("Dimanções na matrix: " + str(np.shape(matrix)))
    f = open('saida_passo_01', 'w')
    for line in matrix:
        for value in line:
            f.write(str(value) + "\t")
        f.write('\n')
    f.close()
    print("Matriz salva no arquivo.")


def print_pygame(matrix_results, access_point):
    """
    Método responsável por desenhar a simulação usando o PyGame.
    :param matrix_results: Matriz float contendo os resultados da simulação.
    :param access_point: Posição (x, y) do ponto de acesso.
    :return: None.
    """
    matrix_max_value = matrix_results.max()
    matrix_min_value = matrix_results.min()

    # print("Desenhando simulação com PyGame...")

    # Lê os valores da matriz que contêm valores calculados e colore
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = get_color_of_interval(matrix_min_value, matrix_max_value, matrix_results[x][y])
            draw_point(color, x, y)

    # Pinta de vermelho a posição do Access Point
    ap = access_point
    draw_point(RED, ap[0], ap[1])

    # draw_floor_plan(floor_plan)

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


def get_color_of_interval(min, max, x):
    """
    Este método retorna uma cor de acordo com o valor que está entre o intervalo min-max. Em outras palavras,
    este método transforma um número em uma cor dentro de uma faixa informada.
    :param min: Valor mínimo do intervalo.
    :param max: Valor máximo do intervalo.
    :param x: Valor que está dentro do intervalo e que deseja saber sua cor.
    :return: Retorna uma tupla representando um cor no formato RGB.
    """
    percentage = get_percentage_of_range(min, max, x)
    color = get_value_in_list(percentage, COLORS)
    # print('Color: ' + str(color))
    return color


def objective_function(matrix):
    """
    Função objetivo para a avaliação da solução atual.
    :param matrix: Matriz a ser avaliada.
    :return: Retorna a soma de todos os elementos da metriz.
    """
    # g = 0
    # for line in matrix:
    #     for value in line:
    #         g += value

    return abs(np.sum(matrix))


propagation_model_gpu = cuda.jit(device=True)(propagation_model)


@cuda.jit  ## mandel_kernel
def simulate_kernel(access_point, matrix):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param save_matrix: Flag que sinaliza se quero salvar a matriz de resultados em um arquivo.
    :param show_pygame: Flag que sinaliza se quero exibir o resultado gráfico da simulação usando o PyGame.
    :param access_point: Access Point com a sua posição.
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """
    # if debug:
    #     print("Iniciando simulação.")

    # Marca o inicio da simulação
    # inicio = datetime.now()


    # if False:
    #     # Inicia o PyGame
    #     pygame.init()
    #
    #     # Configura o tamanho da janela
    #     DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    #     pygame.display.set_caption('Simulando...')

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    # propagation_model_gpu = cuda.jit(device=True)(propagation_model)

    for x in range(startX, WIDTH, gridX):
        for y in range(startY, HEIGHT, gridY):
            value = propagation_model_gpu(x, y, access_point)
            matrix[x][y] = value

            # Guarda os valores máximo e mínimo da matriz ## de fora do metodo
            # matrix_max_value = matrix_results.max()
            # matrix_min_value = matrix_results.min()

            # if show_pygame:
            # Desenha a matriz usando o PyGame
            # print_pygame(matrix_results, access_point)

            # if save_matrix:
            #     # Grava os valores da matriz no arquivo
            #     print_matriz(matrix_results)
            #
            # if show_pygame:
            #     # Atualiza o titulo da janela do PyGame
            #     pygame.display.set_caption('Simulação terminada')
            #
            # if debug:
            #     # Marca o fim da simulação
            #     fim = datetime.now()
            #     # Exibe um resumo da simulação
            #     print('Simulação terminada.')
            #     print("\nInicio: \t" + str(inicio.time()))
            #     print("Fim: \t\t" + str(fim.time()))
            #     print("Duração: \t" + str((fim - inicio).seconds) + " segundos.\n")
            #
            #     print("Maior valor da matriz: " + str(matrix_max_value))
            #     print("Menor valor da matriz: " + str(matrix_min_value))

            # input('\nPrecione qualquer tecla para encerrar a aplicação.')

            ##return matrix_results

@jit
def simulate(access_point, matrix_results):
    """
    Método responsável por realizar a simulação do ambiente de acordo com a posição do Access Point.
    :param save_matrix: Flag que sinaliza se quero salvar a matriz de resultados em um arquivo.
    :param show_pygame: Flag que sinaliza se quero exibir o resultado gráfico da simulação usando o PyGame.
    :param access_point: Access Point com a sua posição.
    :return: Retorna a matriz NxM contendo o resultado da simulação de acordo com o modelo de propagação.
    """

    for x in range(WIDTH):
        for y in range(HEIGHT):
            matrix_results[x][y] = propagation_model(x, y, access_point)

    return matrix_results

#@jit
def get_point_in_circle(point, ray, round_values=True, num=1, absolute_values=True, debug=False):
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

    x = r * np.cos(t) + point[0]
    y = r * np.sin(t) + point[1]

    # Converte todos os valores negativos da lista em positivos
    if absolute_values:
        x = [abs(k) for k in x]
        y = [abs(k) for k in y]

    # if debug:
    #     plt.plot(x, y, "ro", ms=1)
    #     plt.axis([-15, 15, -15, 15])
    #
    #     for i in range(num):
    #         print("Distância entre o ponto ({}, {}) "
    #               "e o ponto ({}, {}) com raio [{}] = {}".format(point[0], point[1], x[i], y[i], ray,
    #                                                              calc_distance(point[0], point[1], x[i],
    #                                                                                 y[i])))
    #     plt.show()

    if round_values:
        x = [round(k) for k in x]
        y = [round(k) for k in y]

    # Verifica se o retorno será um ponto único ou uma lista de pontos.
    if num == 1:
        return [x[0], y[0]]
    else:
        return [x, y]


def starting_temperature():
    """
    Função que calcula a temperatura inicial;
    :return:
    """
    return 100


# @jit
def perturb(S):
    """
     Função que realiza uma perturbação na Solução S.
     Solução pode ser perturbada em um raio 'r' dentro do espaço de simulação.
    :param S: Ponto atual.
    :return: Retorna um ponto dentro do raio informado.
    """
    # Obtem um ponto aleatorio em um raio de X metros
    return get_point_in_circle(point=S, ray=10)


# @jit
def f_cuda(x):
    """
    Valor da função objetivo correspondente á configuração x;
    :param x: Ponto para realizar a simulação.
    :return: Retorna um numero float representando o valor da situação atual.
    """

    # Cria uma matriz para guardar os resultados calculados
    ram_matrix = np.zeros(shape=(WIDTH, HEIGHT), dtype=np.float64)
    # matrix_results = np.zeros(shape=(int(largura_planta / precisao), int(comprimento_planta / precisao)))

    ## 384 threads
    block_dim = (48, 8)
    grid_dim = (int(WIDTH / 8), int(HEIGHT / 48))

    gpu_matrix = cuda.to_device(ram_matrix)

    simulate_kernel[grid_dim, block_dim](x, gpu_matrix)
    gpu_matrix.to_host()

    goal = objective_function(ram_matrix)

    # print("Função objetivo: " + str(goal))
    # print(str((datetime.now() - inicio).seconds) + " segundos ==> por f(x)")

    # contador_uso_func_objetivo += 1

    return goal

def f(x):
    """
    Valor da função objetivo correspondente á configuração x;
    :param x: Ponto para realizar a simulação.
    :return: Retorna um numero float representando o valor da situação atual.
    """

    ram_matrix = np.zeros(shape=(WIDTH, HEIGHT), dtype=np.float64)

    matrix_result = simulate(x, ram_matrix)

    goal = objective_function(matrix_result)

    return goal


def randomize():
    """
    Função que gera um número aleatório no intervalo [0,1];
    :return:
    """

    rand = random()

    # print("Randomiza: " + str(rand))

    return rand


def showSolution(S, matrix):
    print_pygame(simulate_kernel(S, matrix), S)
    draw_floor_plan(floor_plan)


def simulated_annealing(S0, M, P, L, T0, alpha, debug=False):
    """

    :param T0: Temperatura inicial.
    :param S0: Configuração Inicial (Entrada) -> Ponto?.
    :param M: Número máximo de iterações (Entrada).
    :param P: Número máximo de Perturbações por iteração (Entrada).
    :param L: Número máximo de sucessos por iteração (Entrada).
    :param alpha: Factor de redução da temperatura (Entrada).
    :return: Retorna um ponto sendo o mais indicado.
    """
    S = S0
    T = T0
    j = 1

    if debug:
        print("\nIniciando Simulated Annealing com a seguinte configuração:")
        print("Ponto inicial:\t\t\t\t\t\t\t\t" + str(S0))
        print("Númeto máximo de iterações:\t\t\t\t\t" + str(M))
        print("Número máximo de pertubações por iteração:\t" + str(P))
        print("Número máximo de sucessos por iteração:\t\t" + str(L))
        print("Decaimento da teperatura com α=\t\t\t\t" + str(alpha))
        # input("Aperte qualquer tecla para continuar.")

    fS = f(S0)

    # Loop principal – Verifica se foram atendidas as condições de termino do algoritmo
    while True:
        i = 1
        nSucesso = 0

        # Loop Interno – Realização de perturbação em uma iteração
        while True:

            # Tera que mandar o ponto atual e a matriz (certeza?) tbm. Realiza a seleção do ponto.
            Si = perturb(S)
            fSi = f(Si)

            # showSolution(Si)
            print("[\t" + (str(round((100 - 100 * fSi / fS) * 100, 1))) + "\t] S: " + str(S) + "\t Si: " + str(Si))

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
                print("melhor S: " + str(S))

            i = i + 1

            if (nSucesso >= L) or (i > P):
                break

        print("iteração: " + str(j))
        print("temperat: " + str(T) + "\n")

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


# def ccw( A, B, C):
#     return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# @cuda.jit

# @numba.jit("uint8(uint16[:,:], uint16[:,:], uint16[:,:], uint16[:,:])", target='cpu')
def intersect(A, B, C, D):
    # return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    if ((D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])) \
            != ((D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])) \
            and ((C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])) \
                    != ((D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])):
        return 1


########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################
if __name__ == '__main__':

    """
    Contrutor da classe Placement.
    """
    # WIDTH = get_monitor_size()[0] - 100  # Retira 100pxs para folga
    # HEIGHT = get_monitor_size()[1] - 100  # Retira 100pxs para folga

    contador_uso_func_objetivo = 0

    floor_plan = read_walls_from_dxf("/home/samuel/PycharmProjects/TCC/DXFs/bloco-A-l.dxf")

    ## print(numba.typeof(floor_plan))


    heat_map = None

    WIDTH = 384
    HEIGHT = 288

    comprimento_planta = 800
    largura_planta = 600
    precisao = 1  # metro

    escala = HEIGHT / largura_planta

    # tamanho da matriz = dimensão da planta / precisão

    proporcao_planta = comprimento_planta / largura_planta
    # WIDTH = int(HEIGHT * proporcao_planta)


    CHANNEL = 9

    # Posição menor -> Azul, Posição Maior -> Amarelo/Vermelho
    COLORS = [
        '#0C0786', '#100787', '#130689', '#15068A', '#18068B', '#1B068C', '#1D068D', '#1F058E',
        '#21058F', '#230590', '#250591', '#270592', '#290593', '#2B0594', '#2D0494', '#2F0495',
        '#310496', '#330497', '#340498', '#360498', '#380499', '#3A049A', '#3B039A', '#3D039B',
        '#3F039C', '#40039C', '#42039D', '#44039E', '#45039E', '#47029F', '#49029F', '#4A02A0',
        '#4C02A1', '#4E02A1', '#4F02A2', '#5101A2', '#5201A3', '#5401A3', '#5601A3', '#5701A4',
        '#5901A4', '#5A00A5', '#5C00A5', '#5E00A5', '#5F00A6', '#6100A6', '#6200A6', '#6400A7',
        '#6500A7', '#6700A7', '#6800A7', '#6A00A7', '#6C00A8', '#6D00A8', '#6F00A8', '#7000A8',
        '#7200A8', '#7300A8', '#7500A8', '#7601A8', '#7801A8', '#7901A8', '#7B02A8', '#7C02A7',
        '#7E03A7', '#7F03A7', '#8104A7', '#8204A7', '#8405A6', '#8506A6', '#8607A6', '#8807A5',
        '#8908A5', '#8B09A4', '#8C0AA4', '#8E0CA4', '#8F0DA3', '#900EA3', '#920FA2', '#9310A1',
        '#9511A1', '#9612A0', '#9713A0', '#99149F', '#9A159E', '#9B179E', '#9D189D', '#9E199C',
        '#9F1A9B', '#A01B9B', '#A21C9A', '#A31D99', '#A41E98', '#A51F97', '#A72197', '#A82296',
        '#A92395', '#AA2494', '#AC2593', '#AD2692', '#AE2791', '#AF2890', '#B02A8F', '#B12B8F',
        '#B22C8E', '#B42D8D', '#B52E8C', '#B62F8B', '#B7308A', '#B83289', '#B93388', '#BA3487',
        '#BB3586', '#BC3685', '#BD3784', '#BE3883', '#BF3982', '#C03B81', '#C13C80', '#C23D80',
        '#C33E7F', '#C43F7E', '#C5407D', '#C6417C', '#C7427B', '#C8447A', '#C94579', '#CA4678',
        '#CB4777', '#CC4876', '#CD4975', '#CE4A75', '#CF4B74', '#D04D73', '#D14E72', '#D14F71',
        '#D25070', '#D3516F', '#D4526E', '#D5536D', '#D6556D', '#D7566C', '#D7576B', '#D8586A',
        '#D95969', '#DA5A68', '#DB5B67', '#DC5D66', '#DC5E66', '#DD5F65', '#DE6064', '#DF6163',
        '#DF6262', '#E06461', '#E16560', '#E26660', '#E3675F', '#E3685E', '#E46A5D', '#E56B5C',
        '#E56C5B', '#E66D5A', '#E76E5A', '#E87059', '#E87158', '#E97257', '#EA7356', '#EA7455',
        '#EB7654', '#EC7754', '#EC7853', '#ED7952', '#ED7B51', '#EE7C50', '#EF7D4F', '#EF7E4E',
        '#F0804D', '#F0814D', '#F1824C', '#F2844B', '#F2854A', '#F38649', '#F38748', '#F48947',
        '#F48A47', '#F58B46', '#F58D45', '#F68E44', '#F68F43', '#F69142', '#F79241', '#F79341',
        '#F89540', '#F8963F', '#F8983E', '#F9993D', '#F99A3C', '#FA9C3B', '#FA9D3A', '#FA9F3A',
        '#FAA039', '#FBA238', '#FBA337', '#FBA436', '#FCA635', '#FCA735', '#FCA934', '#FCAA33',
        '#FCAC32', '#FCAD31', '#FDAF31', '#FDB030', '#FDB22F', '#FDB32E', '#FDB52D', '#FDB62D',
        '#FDB82C', '#FDB92B', '#FDBB2B', '#FDBC2A', '#FDBE29', '#FDC029', '#FDC128', '#FDC328',
        '#FDC427', '#FDC626', '#FCC726', '#FCC926', '#FCCB25', '#FCCC25', '#FCCE25', '#FBD024',
        '#FBD124', '#FBD324', '#FAD524', '#FAD624', '#FAD824', '#F9D924', '#F9DB24', '#F8DD24',
        '#F8DF24', '#F7E024', '#F7E225', '#F6E425', '#F6E525', '#F5E726', '#F5E926', '#F4EA26',
        '#F3EC26', '#F3EE26', '#F2F026', '#F2F126', '#F1F326', '#F0F525', '#F0F623', '#EFF821'
    ]

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    # print("W" + str(WIDTH))
    # print("H" + str(HEIGHT))

    # Inicia o PyGame
    # pygame.init()

    # Configura o tamanho da janela
    # DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

    if True:
        # access_point = [0, 0]
        # m = p.simulate_kernel(save_matrix=True, show_pygame=True, access_point=access_point)
        # print(p.objective_function(matrix=m))

        access_point = [0, 0]
        ## sugestão: sortear o X,Y do ponto inicial (dentro da matriz)

        ## fixo, procurar uma fórmula para definir o max_iter em função do tamanho da matriz (W*H)
        max_inter = 600

        ## p
        max_pertub = 5

        ## v
        num_max_succ = 80

        ## a
        alpha = .859

        ## t
        temp_inicial = 300

        # Marca o tempo do inicio da simulação
        inicio = datetime.now()

        point = simulated_annealing(S0=access_point, M=max_inter, P=max_pertub, L=num_max_succ, T0=temp_inicial,
                                    alpha=alpha, debug=True)

        # Marca o tempo do fim da simulação
        fim = datetime.now()

        time_seconds = (fim - inicio).seconds
        time_minutes = time_seconds / 60

        print("\nInicio: \t" + str(inicio.time()))
        print("Fim: \t\t" + str(fim.time()))
        print("Duração: \t" + str(time_seconds) + " segundos (" + str(round(time_minutes, 2)) + " minutos).\n")

        print("Melhor ponto sugerido pelo algoritmo: " + str(point))
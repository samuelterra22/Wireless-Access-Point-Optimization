#   https://am207.github.io/2017/wiki/lab4.html
#   https://pt.wikipedia.org/wiki/Simulated_annealing
#   https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
#   https://stackoverflow.com/questions/19720445/invert-negative-values-in-a-list
#   https://www.ime.usp.br/~ghaeser/Hae_Gom.pdf

from math import sqrt, exp
from random import random

import matplotlib.pyplot as plt
import numpy as np


def get_points_in_circle(ray, point, round_values=True, num=1, absolute_values=True, debug=False):
    """
    Método por retorna um ponto ou conjunto de pontos dentro de um determinado raio de um ponto.
    :param ray: Valor do raio desejado.
    :param point: Ponto contendo posição [x, y] de referência do ponto.
    :param round_values: Flag que informa se o(s) ponto(s) serão arredondados. Geralmente será usando para retornar
    valores discretos para posições da matriz.
    :param absolute_values: Flag que informa se o(s) ponto(s) serão absolutos (positivos).
    :param num: Número de pontos que deseja gerar. Gera um ponto como default.
    :param debug: Flag que quando informada True, printa na tela o(s) ponto(s) gerados e a distância do ponto de
    referência.
    :return: Um ponto ou um conjunto de pontos do tipo float
    """

    t = np.random.uniform(0.0, 2.0 * np.pi, num)
    r = ray * np.sqrt(np.random.uniform(0.0, 1.0, num))

    x = r * np.cos(t) + point[0]
    y = r * np.sin(t) + point[1]

    # Converte todos os valores negativos da lista em positivos
    if absolute_values:
        x = [abs(k) for k in x]
        y = [abs(k) for k in y]

    if debug:
        plt.plot(x, y, "ro", ms=1)
        plt.axis([-15, 15, -15, 15])

        for i in range(num):
            print("Distância entre o ponto ({}, {}) "
                  "e o ponto ({}, {}) com raio [{}] = {}".format(point[0], point[1], x[i], y[i], ray,
                                                                 calc_distance(point[0], point[1], x[i], y[i])))
        plt.show()

    if round_values:
        x = [round(k) for k in x]
        y = [round(k) for k in y]

    # Verifica se o retorno será um ponto único ou uma lista de pontos.
    if num == 1:
        return [x[0], y[0]]
    else:
        return [x, y]


def temperatura_inicial():
    """
    Função que calcula a temperatura inicial;
    :return:
    """
    return 100


def pertuba(S, matrix):
    """
     Função que realiza uma perturbação na Solução S;
     Solução pode ser perturbada em um raio 'r' dentro do espaço de simulação
    :param S:
    :return:
    """
    return 0


def f(x):
    """
    Valor da função objetivo correspondente á configuração x;
    :param x:
    :return:
    """
    return 0


def randomiza():
    """
    Função que gera um número aleatório no intervalo [0,1];
    :return:
    """
    return random.random()

def simulated_annealing(S0, M, P, L, alpha, matrix):
    """

    :param S0: Configuração Inicial (Entrada) -> Ponto?;
    :param M: Número máximo de iterações (Entrada);
    :param P: Número máximo de Perturbações por iteração (Entrada);
    :param L: Número máximo de sucessos por iteração (Entrada);
    :param alpha: Factor de redução da temperatura (Entrada);
    :return:
    """
    S = S0
    T0 = temperatura_inicial()   # Pode ser passado por paramentro?
    T = T0
    j = 1

    # Loop principal – Verifica se foram atendidas as condições de termino do algoritmo
    while True:
        i = 1
        nSucesso = 0

        # Loop Interno – Realização de perturbação em uma iteração
        while True:

            Si = pertuba(S, matrix)         # Tera que mandar o ponto atual e a matriz tbm. Realiza a simulação
            deltaFi = f(Si) - f(S)  # Verificar se o retorno da função objetivo está correto.

            # Teste de aceitação de uma nova solução
            if (deltaFi <= 0) or (exp(-deltaFi / T) > randomiza()):
                S = Si
                nSucesso = nSucesso + 1

            i = i + 1

            if (nSucesso >= L) or (i > P):
                break

        # Atualização da temperatura (Deicaimento geométrico)
        T = alpha * T

        # Atualização do contador de iterações
        j = j + 1

        if (nSucesso == 0) or (j > M):
            break

    return S


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


if __name__ == '__main__':

    print(get_points_in_circle(3, [0, 0]))

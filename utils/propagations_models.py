import math
from math import log10
from numba import jit


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
def two_par_logistic(Pt_dBm, x):
    # https://en.wikipedia.org/wiki/Logistic_distribution#Related_distributions
    return Pt_dBm - (-15.11596 * math.log10(x * 2.1642))


@jit
def four_par_log(Pt_dBm, x):
    A = 79.500
    B = -38
    C = -100.000
    D = 0.0
    E = 0.005

    # https://en.wikipedia.org/wiki/Shifted_log-logistic_distribution
    return Pt_dBm - (D + (A - D) / (pow((1 + pow((x / C), B)), E)))


@jit
def five_par_log(Pt_dBm, x):
    A = 84.0
    B = -48
    C = -121.0
    D = -5.0
    E = 0.005
    # https://en.wikipedia.org/wiki/Shifted_log-logistic_distribution
    return Pt_dBm - (D + (A - D) / (pow((1 + pow((x / C), B)), E)))

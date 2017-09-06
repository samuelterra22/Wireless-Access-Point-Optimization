# Ler/Escrever arquivo em python
# http://www.pythonforbeginners.com/files/reading-and-writing-files-in-python
#
#

from datetime import datetime

from Placement import Placement

def p_f_2_k():
    # Vizinhos
    A = [80, 160]

    # Temperatura Inicial
    B = [300, 600]

    # Fator de esfriamento
    C = [.80, .85]

    # Pertubações
    D = [5, 10]

    # Tabela verdade com as configurações das possibilidades. Os valores representam a posição dos vetores de parametros.
    CONFIGURACOES = [
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]

    p = Placement()

    # fixo
    access_point = [0, 0]

    # fixo
    max_inter = 50

    # Cria o arquivo de saida
    print("Gerando arquivo de saida.")
    f = open('saida_2k', 'w')
    f.write("i\tPonto\tIterações\tVizinhos\tT0\tAlfa\tPertubações\tInicio\tFim\tDuração(Seg)\tDuração(Min)\tF.O.\n")
    f.write("-\t-----\t---------\t--------\t--\t----\t-----------\t------\t---\t------------\t------------\t----\n")
    f.close()

    # contador usado apenas no arquivo de saida
    i = 0

    dura_seg = []
    dura_min = []
    f_o = []

    for line in CONFIGURACOES:
        # Abre o aquivo em modo 'append'
        f = open('saida_2k', 'a')

        print("Iteração: " + str(i))

        ## vizinhos - A
        num_max_succ = A[line[0]]

        ## Temperatura inicial - B
        temp_inicial = B[line[1]]

        ## fator de resfriamento - C
        alpha = C[line[2]]

        ## pertubacoes - D
        max_pertub = D[line[3]]

        print("Iniciando simulação.")

        # Marca o tempo do inicio da simulação
        inicio = datetime.now()

        # Realiza a busca do melhor ponto
        point = p.simulated_annealing(S0=access_point, M=max_inter, P=max_pertub, L=num_max_succ, T0=temp_inicial,
                                      alpha=alpha, debug=True)

        # Marca o tempo do fim da simulação
        fim = datetime.now()

        time_seconds = (fim - inicio).seconds
        time_minutes = time_seconds / 60

        fo = p.f(point)

        print("\nInicio: \t" + str(inicio.time()))
        print("Fim: \t\t" + str(fim.time()))
        print("Duração: \t" + str(time_seconds) + " segundos (" + str(round(time_minutes, 2)) + " minutos).\n")
        print("Valor da função objetivo: \t" + str(fo))

        print("i\tPonto\tIterações\tVizinhos\tT0\tAlfa\tPertubações\tInicio\tFim\tDuração(Seg)\tDuração(Min)\tF.O.\n")
        print(str(i) + "\t" + str(access_point) + "\t" + str(max_inter) + "\t" + str(num_max_succ) + "\t" +
              str(temp_inicial) + "\t" + str(alpha) + "\t" + str(max_pertub) + "\t" + str(inicio) + "\t" + str(fim) +
              "\t" + str(time_seconds) + "\t" + str(round(time_minutes, 2)) + "\t" + str(fo) + "\n")

        f.write(str(i) + "\t" + str(access_point) + "\t" + str(max_inter) + "\t" + str(num_max_succ) + "\t" +
                str(temp_inicial) + "\t" + str(alpha) + "\t" + str(max_pertub) + "\t" + str(inicio) + "\t" + str(fim) +
                "\t" + str(time_seconds) + "\t" + str(round(time_minutes, 2)) + "\t" + str(fo) + "\n")

        # Adiciona resultados em listas
        dura_seg.append(time_seconds)
        dura_min.append(time_minutes)
        f_o.append(fo)

        i = i + 1

        # Fecha o arquivo
        f.close()

    results = {
        "dura_seg": dura_seg,
        "dura_min": dura_min,
        "f_o": f_o
    }

    print("Experimento Fatorial 2K terminado.")
    return results

def print_results_2k(values):

    seconds = values.get('dura_seg')
    minutes = values.get('dura_min')
    fo = values.get('f_o')

    print("F.O.\tSegundos\tMinutos\n")
    for i in range(len(seconds)):
        print(str(fo[i]) + "\t"+str(round(seconds[i], 2)) + "\t"+str(minutes[i]))

    print("-------------------------------------")
    print("Maior valor da F.O.:\t" + str(max(fo)))
    print("Menor valor da F.O.:\t" + str(min(fo)))
    print("-------------------------------------")
    print("Maior valor do tempo (seg.):\t" + str(max(seconds)))
    print("Menor valor do tempo (seg.):\t" + str(min(seconds)))
    print("-------------------------------------")
    print("Maior valor do tempo (min.):\t" + str(round(max(minutes), 3)))
    print("Menor valor do tempo (min.):\t" + str(round(min(minutes), 3)))
    print("-------------------------------------")

values = p_f_2_k()

print_results_2k(values)
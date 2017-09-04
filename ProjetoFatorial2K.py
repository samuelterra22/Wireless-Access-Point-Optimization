from datetime import datetime

from Placement import Placement

# Vizinhos
A = [20, 40]

# Temperatura Inicial
B = [100, 200]

# Fator de esfriamento
C = [.75, .95]

# Pertubações
D = [10, 20]

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

##fixo
access_point = [0, 0]

##fixo
max_inter = 1000

# Cria o arquivo de saida
print("Gerando arquivo de saida.")
f = open('saida_2k', 'w')
f.write("i\tPonto\tIterações\tVizinhos\tT0\tAlfa\tPertubações\tInicio\tFim\tDuração(Seg)\tDuração(Min)\tF.O.\n")
f.write("-\t-----\t---------\t--------\t--\t----\t-----------\t------\t---\t------------\t------------\t----\n")
f.close()

# contato usado apenas no arquivo de saida
i = 0

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
          "\t" + str(time_seconds) + "\t" + str(time_minutes) + "\t" + str(fo) + "\n")

    f.write(str(i) + "\t" + str(access_point) + "\t" + str(max_inter) + "\t" + str(num_max_succ) + "\t" +
            str(temp_inicial) + "\t" + str(alpha) + "\t" + str(max_pertub) + "\t" + str(inicio) + "\t" + str(fim) +
            "\t" + str(time_seconds) + "\t" + str(time_minutes) + "\t" + str(fo) + "\n")

    i = i + 1

    # Fecha o arquivo
    f.close()

print("Experimento Fatorial 2K terminado.")

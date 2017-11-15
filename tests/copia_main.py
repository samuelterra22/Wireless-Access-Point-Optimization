if __name__ == '__main__':

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    ##################################################
    #  CONFIGURAÇÕES DOS EQUIPAMENTOS

    # OBS.: por conta da precisao de casas decimais do float
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
    dbm_absorvido_por_parede = 8

    # Potência de transmissão de cada AP
    # Pt_dBm = -14
    # Pt_dBm = -17
    # Pt_dBm = -20
    Pt_dBm = -25
    # Pt_dBm = -30

    # Quantidade de APs
    num_aps = 2

    POSICAO_INICIAL_ALEATORIA = False

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

    # ENVIRONMENT = "GPU"
    ENVIRONMENT = "CPU"

    # Tamanho da simulação
    TAMAMHO_SIMULACAO = 400
    # TAMAMHO_SIMULACAO = 600

    # Ativa / Desativa a animação passo a passo da otimização
    # ANIMACAO_PASSO_A_PASSO   = True
    ANIMACAO_PASSO_A_PASSO = False

    # ANIMACAO_MELHORES_LOCAIS = True
    ANIMACAO_MELHORES_LOCAIS = False

    # ANIMACAO_MELHORES = True
    ANIMACAO_MELHORES = False

    ##################################################

    # Lista para guardar as funções objetivos calculadas durante a simulação
    FOs = []

    WIDTH = TAMAMHO_SIMULACAO
    HEIGHT = int(WIDTH * (largura_planta / comprimento_planta))
    escala = WIDTH / comprimento_planta
    precisao = COMPRIMENTO_EDIFICIO / WIDTH

    TOTAL_PONTOS = WIDTH * HEIGHT

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
    max_inter = 600
    # max_inter = 600 * (1 + num_aps)
    # max_inter = 600 * (10 * num_aps)
    # max_inter = TOTAL_PONTOS * 0.2

    # p - Máximo de perturbações
    max_pertub = 5

    # RAIO_PERTURBACAO = WIDTH * 0.0100
    # RAIO_PERTURBACAO = WIDTH * 0.0175
    # RAIO_PERTURBACAO = WIDTH * 0.0250
    # RAIO_PERTURBACAO = WIDTH * 0.1100
    beta = 1
    RAIO_PERTURBACAO = (1 / precisao) * (beta + num_aps)  ## VALADAO testing
    # RAIO_PERTURBACAO = (1 / precisao) * (1 + num_aps)  ## VALADAO testing

    # v - Máximo de vizinhos
    # num_max_succ = 80
    # num_max_succ = 80 * 10
    # num_max_succ = 80 * (beta + num_aps) * 3
    num_max_succ = 240 * (beta + num_aps)

    # a - Alpha
    alpha = .85
    # alpha = .95

    # t - Temperatura
    temp_inicial = 300 * (beta + num_aps)
    # temp_inicial = 300 * (1 + num_aps) * 10

    # Máximo de iterações do S.A.
    max_SA = 1
    ##################################################

    # Visualização dos dados
    # Inicia o PyGame e configura o tamanho da janela
    # pygame.init()
    # icon = pygame.image.load('images/icon.png')
    # pygame.display.set_icon(icon)
    # pygame.display.set_caption("Resultado Simulação - IFMG Campus Formiga")
    # DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

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

    # input('\nAperte ESC para fechar a simulação.')

    # profile.runctx('run()', globals(), locals(),'tese')
    # cProfile.run(statement='run()', filename='PlacementAPs.cprof')

    # python ../PlacementAPs.py | egrep "(tottime)|(PlacementAPs.py)" | tee ../cProfile/PlacementAPs.py_COM-JIT.txt
    # cat ../cProfile/PlacementAPs.py_COM-JIT.txt | sort -k 2 -r

    # python PlacementAPs.py | egrep '(ncalls)|(PlacementAPs)'
    # https://julien.danjou.info/blog/2015/guide-to-python-profiling-cprofile-concrete-case-carbonara

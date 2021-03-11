#%%
import random
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import data_generate_process
from record import Record
from copy import deepcopy
import numexpr as ne
import transform_result as transform
#%%

class GlobalParameter:
    def __init__(self):
        self.col_names = ["x1", 'x2', 'y1']
        self.pop_size = 500
        self.tournament_size = 3
        self.df = pd.DataFrame(data=[[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["x1", 'x2', 'y1'])
        self.EN_ridge_ratio = 0.5
        self.EN_lamda = 1 # which from sklearn
        self.SCAD_a = 3.7
        self.SCAD_lamda = 0.5
        self.MCP_a = 3.7
        self.MCP_lamda = 0.5


GLOBAL = GlobalParameter()
#%%
## express coe
def envalue(encode, start, sep):
    recode = encode[:]
    recode.reverse()
    for i in range(len(recode)):
        # print(recode[i])
        if recode[i]:
            # print(start)
            start += sep* 2**i
    return start
#%%
COE_NAME = ["coe_x1x2", "coe_x2", "coe_x1", "coe_y1", "coe_b"]
n_COE = len(COE_NAME)
INDEX = ["a", "n"]
COES_pd = pd.DataFrame()
## sci_coe: a* 10**n
## a [1, 10)
## n [-4, 3]
ALPHA = ["alpha", "beta"]
#%%
## encoding
# 6 slot for n in sci_coe, first digit indicate +-, ecah represent 0.29, a +-[1, 10), max=9.99  
# 3 slot for n in sci_coe, ecah represent 1, n [-4, 3]
# 3 slot for alpha, beta, ecah represent 0.1, alpha [0.1, 0.8]
SLOT_a = 6
SLOT_n = 3
SLOT_alpha = 3

START_a = 1
START_n = -4
START_alpha = 0.1

SEP_a = 0.29
SEP_n = 1
SEP_alpha = 0.1

encode_As = []
encode_Ns = []

def do_encode_alphas():
    temp_encode_alphas = []
    for i in range(2):
        temp_encode_alphas.append([random.choice((0, 1)) for _ in range(SLOT_alpha)])
    return temp_encode_alphas

## initalize till alpha beta valid
encode_alphas = do_encode_alphas()
while envalue(encode=encode_alphas[0], start=START_alpha, sep=SEP_alpha) + envalue(encode=encode_alphas[1], start=START_alpha, sep=SEP_alpha) > 1:
    encode_alphas = do_encode_alphas()


for i in range(n_COE):
    encode_As.append([random.choice((0, 1)) for _ in range(SLOT_a)])
    encode_Ns.append([random.choice((0, 1)) for _ in range(SLOT_n)])
#%%
# print(envalue(encode=encode_Ns[1], start=START_n, sep=SEP_n))
# print(envalue(encode=[1, 1, 1], start=START_n, sep=SEP_n))
# print(envalue(encode=[1, 1, 1, 1, 1], start=START_a, sep=SEP_a))

def coe_value(encode_A, encode_N):
    sign = 1
    if encode_A[0]:
        sign = -1
    a = envalue(encode=encode_A[1:], start=START_a, sep=SEP_a)
    n = envalue(encode=encode_N, start=START_n, sep=SEP_n)
    return sign * a * 10 ** n

for i in range(n_COE):
    print(round(coe_value(encode_A=encode_As[i], encode_N=encode_Ns[i]), 2))
#%%
## crossover
def do_crossover(par1, par2, method="single_point"):
    gen_len = len(par1)
    cross_point = random.randrange(gen_len * 2)
    # cross_point = 8
    if cross_point // gen_len:
        temp = par1
        par1 = par2
        par2 = temp
    off1 = par1[:cross_point] + par2[cross_point:]
    off2 = par2[:cross_point] + par1[cross_point:]
    return off1, off2

encode_As[0], encode_As[1] = do_crossover(par1=encode_As[0], par2=encode_As[1])

#%%
## mutation
def do_mutation(parent, mutation_rate=0.1, method="each_point"):
    # gen_len = len(parent)
    offspring = []
    for point in parent:
        if mutation_rate > random.random():
            point += 1
        offspring.append(point % 2)
    return offspring

#%%
def get_coes(encode_As, encode_Ns):
    coes = []
    for i in range(n_COE):
        coes.append(round(coe_value(encode_A=encode_As[i], encode_N=encode_Ns[i]), 3))
    return coes
#%%
def get_alphas(encode_alphas):
    alphas = []
    for i in range(2):
        alphas.append(round(envalue(encode=encode_alphas[i], start=START_alpha, sep=SEP_alpha), 1))
    return alphas

#%%
## express
def program_express(encode_As, encode_Ns, encode_alphas):
    coes = []
    for i in range(n_COE):
        coes.append(round(coe_value(encode_A=encode_As[i], encode_N=encode_Ns[i]), 3))
    alphas = []
    for i in range(2):
        alphas.append(round(envalue(encode=encode_alphas[i], start=START_alpha, sep=SEP_alpha), 1))
    print("{0[0]}* x1 ** {1[0]} * x2 ** {1[1]} + {0[1]} * x2 + {0[2]} * x1 + {0[3]} * y1 + {0[4]}" .format(coes, alphas))
    return "{0[0]}* x1 ** {1[0]} * x2 ** {1[1]} + {0[1]} * x2 + {0[2]} * x1 + {0[3]} * y1 + {0[4]}" .format(coes, alphas)
#%%
test_pro = program_express(encode_As, encode_Ns, encode_alphas)
#%%
x1 =5
x2 = 4
y1 = 3
print(eval(test_pro))
#%%
## fitness
prediction = round(eval(test_pro), 4)
y_true = 11
def compute_fitness(prediction, y_true, coes):
    rss = np.sum(np.subtract(prediction, y_true.to_list()) ** 2)
    en = np.sum([rss, GLOBAL.EN_lamda * (GLOBAL.EN_ridge_ratio * np.sum(np.power(coes, 2)) + (1-GLOBAL.EN_ridge_ratio) * np.sum(np.abs(coes)))])
    # print("rss", rss)
    # print("coes", coes)
    # print("en", en)
    fitness = en
    return fitness
# print(compute_fitness(prediction=prediction, y_true=y_true, coes=get_coes(encode_As=encode_As, encode_Ns=encode_Ns)))
#%%
class Continous_Encode:
    def __init__(self) -> None:
        encode_alphas = do_encode_alphas()
        while envalue(encode=encode_alphas[0], start=START_alpha, sep=SEP_alpha) + envalue(encode=encode_alphas[1], start=START_alpha, sep=SEP_alpha) > 1:
            encode_alphas = do_encode_alphas()
        encode_As = []
        encode_Ns = []
        for i in range(n_COE):
            encode_As.append([random.choice((0, 1)) for _ in range(SLOT_a)])
            encode_Ns.append([random.choice((0, 1)) for _ in range(SLOT_n)])

        self.encode_alphas = encode_alphas
        self.encode_As = encode_As
        self.encode_Ns = encode_Ns
        pass

    def program_print(self):
        print(self.program_express)
        pass

    @property
    def coes(self):
        return get_coes(encode_As=self.encode_As, encode_Ns=self.encode_Ns)

    @property
    def alphas(self):
        return get_alphas(encode_alphas=self.encode_alphas)

    @property
    def program_express(self):
        return "{0[0]} * x1 ** {1[0]} * x2 ** {1[1]} + {0[1]} * x2 + {0[2]} * x1 + {0[3]} * y1 + {0[4]}" .format(self.coes, self.alphas)
#%%

import requests

def lineNotifyMessage(token, msg):
  headers = {
      "Authorization": "Bearer " + token,
      "Content-Type" : "application/x-www-form-urlencoded"
  }

  payload = {'message': msg}
  r = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=payload)
  return r.status_code
#%%
#%%
DATA, Y_TRUE = data_generate_process.dgp(method="MIMO_1", n=500)
#%%
def experiment(exp_name = "eriment", EN_ridge_ratio=False, EN_lamda=False, SCAD_a=False, SCAD_lamda=False, MCP_a=False, MCP_lamda=False):

    # making directicory if doesn't excist 
    out_dir = "./%s" %(exp_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    t_start = time.time()
    check = False
    exp_Records = []
    MAX_GENERATIONS = 500
    EXP_TIMES = 30
    POP_SIZE = 500
    FITNESS_METHOD = "EN"

    ## setting testing experiment hyper-parameters
    if exp_name == "eriment":
        check = True
        MAX_GENERATIONS = 25
        EXP_TIMES = 10
        POP_SIZE = 10

    ## setting general experiment hyper-parameters
    if EN_lamda:
        GLOBAL.EN_lamda = EN_lamda

    ## experiment starts here
    for i in range(EXP_TIMES):
        print()
        print("-------------------")
        print("%d time experiment" %i)
        print("-------------------")
        print()
        # NEW_POP_PCT = 0.1
        col_name = DATA.columns.to_list()
        population = [Continous_Encode() for _ in range(POP_SIZE)]
        TOURNAMENT_SIZE = 3
        XOVER_PCT = 0.7
        REG_STRENGTH = 2

        global_best = float("inf")
        fitness_list = [float("inf")] * 20
        prog_express = "temp"
        ts = time.time()
        exp_Records.append(Record())
        t1 = time.time()
        for gen in range(MAX_GENERATIONS):
            fitness = []
            # change_flag = 1
            kk=0
            best_prog = population[0]
            for encode in population:
                # print(kk)
                kk += 1
                # prog.program_print()
                prediction = ne.evaluate(encode.program_express, local_dict=DATA).tolist()
                # print(type(prediction))
                # print(prediction)
                score = compute_fitness(prediction=prediction, y_true=Y_TRUE, coes=encode.coes)
                # print(score)
                if np.isnan(score):
                    score = np.inf
                fitness.append(score)
                if np.isinf(score):
                    continue

                if score < global_best:
                    global_best = score
                    best_prog = deepcopy(encode)

            prog_express = best_prog.program_express
            t2 = time.time()
            time_cost = t2-t1
            t1 = time.time()

            ## converge critiria
            fitness_list[gen%20] = global_best
            fitness_var = np.var(fitness_list)

            print()
            print("%d time experiment" % i)
            # print("unchanged_score: %d" %unchanged_score)
            print("Generation: %d" %gen)
            print("fitness_var: %.5f" %fitness_var)
            print("Best Score: %.5f" %global_best)
            print("Median Score: %.5f" %pd.Series(fitness).median())
            print("Best program: %s" %prog_express)
            print("Time used: %d sec\n" %time_cost)

            ## recording
            exp_Records[i].update_all(fitness=global_best,program=prog_express, t=time_cost)
            
            if 0.01 > fitness_var:
                print("\n---\n---\n\nexperiment has converge when fitness varience < 0.01, and best fitness is %f\n\n---\n---\n" %global_best)
                break

            ## next generation
            next_population = []
            for _ in range(int(POP_SIZE/2)):
                # print()
                # print(len(population))
                # print(len(fitness))
                def tournament_select():
                    # randomly select population members for the tournament
                    tournament_members = [random.randint(0, POP_SIZE - 1) for _ in range(TOURNAMENT_SIZE)]
                    # select tournament member with best fitness
                    member_fitness = [(fitness[i], population[i]) for i in tournament_members]
                    return min(member_fitness, key=lambda x: x[0])[1]
                ## find feasible tournament for alpha, beta crossover
                feasible_production_function = False
                while not feasible_production_function:
                    parents = [tournament_select(), tournament_select()]

                    offspring1 = deepcopy(parents[0])
                    offspring2 = deepcopy(parents[1])
                    ## alpha, beta part
                    offspring1.encode_alphas[0], offspring2.encode_alphas[0] = do_crossover(par1=parents[0].encode_alphas[0], par2=parents[1].encode_alphas[0])
                    offspring1.encode_alphas[1], offspring2.encode_alphas[1] = do_crossover(par1=parents[0].encode_alphas[1], par2=parents[1].encode_alphas[1])
                    if offspring1.alphas[0] + offspring1.alphas[1] <= 1:
                        if offspring2.alphas[0] + offspring2.alphas[1] <= 1:
                            feasible_production_function = True

                for j in range(n_COE):
                    offspring1.encode_As[j], offspring2.encode_As[j] = do_crossover(par1=parents[0].encode_As[j], par2=parents[1].encode_As[j])
                    offspring1.encode_Ns[j], offspring2.encode_Ns[j] = do_crossover(par1=parents[0].encode_Ns[j], par2=parents[1].encode_Ns[j])
                    
                    offspring1.encode_As[j] = do_mutation(offspring1.encode_As[j])
                    offspring2.encode_As[j] = do_mutation(offspring2.encode_As[j])
                
                    offspring1.encode_Ns[j] = do_mutation(offspring1.encode_Ns[j])
                    offspring2.encode_Ns[j] = do_mutation(offspring2.encode_Ns[j])
                
                next_population.append(offspring1)
                next_population.append(offspring2)
            next_population.append(best_prog)
            # print(len(population))
            # print("len(next_population): ", len(next_population))
            population = next_population
            # print(len(population))
            if 0 == (gen+1)%100:
                t_final = time.time()
                # 修改為你要傳送的訊息內容
                m = "\nso far time cost: %d sec" % (t_final - t_start)
                message = "\n\nexp_" + str(exp_name) + " not converge yet,\nwith fitness:\n" + str(fitness_list) + m
                # 修改為你的權杖內容
                token = 'CCgjmKSEGamkEj9JvhuIkFNYTrpPKHyCb1zdsYRjo86'
                lineNotifyMessage(token, message)

        tf = time.time()
        print("Best score: %f" % global_best)
        print("Best program: %s" % prog_express)
        print("Total time: %d sec" % (tf - ts))
        ## one experiment finished
    
    ## experiments finished

    ## naming experiment stuff
    plot_name = str(exp_name) + "_fitness_scatter"
    plot_file_name=os.path.join(out_dir, plot_name)
    prog_name = str(exp_name) + "_program"
    prog_file_name=os.path.join(out_dir, prog_name)

    ## plot fitness
    plt.figure(figsize=(20, 12.5))
    for i in range(EXP_TIMES):
        plt.plot(exp_Records[i].generation, exp_Records[i].fitness, label = "%d time experiment" %(i+1)) 
    plt.legend(loc='upper right')
    plt.title("experiment of %s" %exp_name)  # title
    plt.ylabel("fitness")  # y label
    plt.xlabel("iteration")  # x label
    plt.savefig('%s.png' % plot_file_name, dpi=600, format='png')
    plt.show()

    ## save programs
    best_programs = []
    time_used = []
    gen_counts = []
    for i in range(EXP_TIMES):
        best_programs.append(exp_Records[i].best_programs[-1])
        time_used.append(sum(exp_Records[i].time_used))
        gen_counts.append(exp_Records[i].gen_count)

    file = pd.DataFrame(data=[best_programs, time_used, gen_counts], index=["best_programs", "time_used", "gen_counts"]).T
    file.to_csv("%s.csv" % prog_file_name , index=False)

    transform.coe_substract(prog_file_name)
    if not check:
        t_final = time.time()
        # 修改為你要傳送的訊息內容
        m = "\nTotal time: %d sec" % (t_final - t_start)
        message = "\n\nexp_" + str(exp_name) + "complete" + m
        # 修改為你的權杖內容
        token = 'CCgjmKSEGamkEj9JvhuIkFNYTrpPKHyCb1zdsYRjo86'

        lineNotifyMessage(token, message)

    return exp_Records
#%%
expTemp = experiment()

#%%

do_v7_lamda = True
# _ridgeRatio_Lamda
if do_v7_lamda:
    EN_ridge_ratio = 0.5
    lamdas = [0.1, 1, 10, 20, 50, 100]
    exp_v5_lamdas = []
    for lamda in lamdas:
        name = "v7_continuous_lamda_{}" .format (lamda)
        exp_v5_lamdas.append(experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=lamda))
#%%

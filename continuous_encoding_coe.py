#%%
import random
import numpy as np
import pandas as pd
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

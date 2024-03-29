#%%
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.sparse import data
import transform_result as transform
import os
import data_generate_process
from record import Record
#%%
# get data
DATA, Y_TRUE = data_generate_process.dgp(method="MIMO_1", n=500)
#%%
def mimo_1(coes, pars):
    """[summary]

    Args:
        coes (list): [coe_x1x2, alpha, beta, coe_x1, coe_x2, coe_y1, coe_b]
        pars (list): [x1, x2, y1]

    Returns:
        [type]: [y2]
    """
    coe_x1x2 = coes[0]; alpha = coes[1]; beta = coes[2]
    coe_x1 = coes[3]; coe_x2 = coes[4]
    coe_y1 = coes[5]; coe_b = coes[6]
    x1 = pars[0]; x2 = pars[1]; y1 = pars[2]
    return coe_x1x2 * (x1 ** alpha) * (x2 ** beta) + coe_x1 * x1 + coe_x2 * x2 + coe_y1 * y1 + coe_b
#%%
coe_x1x2 = 1.1
alpha = 0.3
beta = 0.4
coe_y1_eff = (-1)

n = 500
input_range_low = 1
input_range_up = 2
output_range_low = 2
output_range_up = 5


def generate_uniform_data(lower, upper, n, col=1):
    return np.random.uniform(lower, upper, n * col).reshape(n, col)

x = generate_uniform_data(input_range_low, input_range_up, n, 2)
y = generate_uniform_data(output_range_low, output_range_up, n, 2)

## 製作 slope
slope = (y.T[1] / y.T[0])

## 參數設定
coe_x1x2 = 1.0845
pow_x1 = 0.3
pow_x2 = 0.4
coe_y1_eff = (-1)

y1_eff = coe_x1x2 * (x.T[0] ** pow_x1) * (x.T[1] ** pow_x2) / (slope + 1)
PARS = [x.T[0], x.T[1], y1_eff]
COES = [coe_x1x2, alpha, beta, 0, 0, coe_y1_eff, 0]
Y2_TRUE = mimo_1(coes=COES, pars=PARS)
#%%
def fun2(coes_init):
    return mimo_1(coes = coes_init, pars= PARS) - Y2_TRUE
#%%
coes_init = [0, 1, 2, 3, 4, 5, 6]
res2 = least_squares(fun=fun2, x0=coes_init)
#%%
class Record:
    def __init__(self, result):
        self.coes = result.x
#%%
res = []
for _ in range(30):
    result = Record(least_squares(fun=fun2, x0=coes_init))
    res.append(result.coes)
#%%
COL_NAMES = ["coe_x1x2", "alpha", "beta", "coe_x1", "coe_x2", "coe_y1", "coe_b"]
final = pd.DataFrame(data=res, columns=COL_NAMES)
#%%
final_des = final.describe().iloc[1].to_list()
for i in range(len(final_des)):
    final_des[i] = round(final_des[i], 5)
#%%
print("coes  [coe_x1x2, alpha, beta, coe_x1, coe_x2, coe_y1, coe_b]")
print("true coes", COES)
print("predicted", res2.x)
#%%
predicted_06 = [0.81, 0.25, 0.37, 0.04, 0.02, -1.1, 13.2]
predicted_07 = [0.8, 0.28, 0.36, 2.58, 0.41, -38.88, 0.01]
predicted_08 = [0.74, 0.26, 0.36, 0.54, 52.19, -4.53, 45.98]
#%%
result = pd.DataFrame(data=[COES, final_des, predicted_06, predicted_07, predicted_08], columns=COL_NAMES, index=["True coes", "predicted_scipy", "predicted_06", "predicted_07", "predicted_08"])

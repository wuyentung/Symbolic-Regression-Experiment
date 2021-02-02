#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
#%%
v21_df = pd.read_csv("expV21_program.txt")
v21_df.head()
#%%
print(type(v21_df.iloc[0][0]))
print(type(v21_df.iloc[0][1]))

#%%
# temp = deepcopy(v2_df.iloc[0][0])
# print(temp)
# %%
def clean_prog(prog):
    """[summary]

    Args:
        prog (str): [description]
    :rtype: list
    :return: [coe_x1x2, alpha, beta, coe_x2, coe_x1, coe_y1, coe_b(with operator)]
    """
    step1 = prog.replace("(", "")
    step2 = step1.replace(")", "")
    step3_list = step2.split(" ")
    single_index = [0, 4, 8, 10, 14]
    compound_index = [17, 21]
    final = []
    for i in range(len(step3_list)):
        if i in single_index:
            final.append(eval(step3_list[i]))
        if i in compound_index:
            final.append(eval(step3_list[i] + step3_list[i+1]))
    return final
# %%
# temp2 = clean_prog(temp)
# %%
coes = []
for i in range(v21_df.shape[0]):
    coes.append(clean_prog(v21_df.iloc[i][0]))
#%%
print(coes)
# %%
coe_column = ["coe_x1x2", "alpha", "beta", "coe_x2", "coe_x1", "coe_y1", "coe_b"]
coes_df = pd.DataFrame(data=coes, columns=coe_column)
coes_df.head()
# %%
coes_df.describe()
# %%
# plt.plot(coes_df)
# %%

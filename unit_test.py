#%%
from copy import deepcopy
import random
import pandas as pd
import tree
import numpy as np
import transform_result as transform
import numexpr as ne
import data_generate_process
import matplotlib.pyplot as plt
#%%
locals = []
max_len = 0
for i in range(30):
    local_limit = random.randint(3, 900)
    locals.append([i] * local_limit)
    if local_limit > max_len:
        max_len = local_limit
print(locals)
#%%
fig = plt.figure(figsize=(32, 16))
ax1 = fig.add_subplot(111)

for i in range(30):
    ax1.scatter(locals[i], range(len(locals[i])),  s=10, marker="X", label='%d' %i)
plt.legend(loc='upper left');
plt.savefig('%s.png' % "temp", dpi=600, format='png')
plt.show()
#%%
# create data 
x1 = [10,20,30,40,50, 60, 75] 
x2 = [10,20,30] 
y1 = [30,30,10,30,30, 5, 5] 
y2 = [30,30,50] 
y = [y1, y2]
x = [x1, x2]
# plot lines 
for i in range(2):
    plt.plot(x[i], y[i], label = i) 
plt.legend() 
plt.show()
#%%
method = "MIMO_1"
data = pd.read_csv("dgp_%s.csv" %method)
data.head()
#%%
data1, y1 = data_generate_process.dgp()
data1.head()
#%%
t1 = tree.tree()
t1.program_express
#%%
tree.evaluate(root=t1)
#%%
tree.evaluate(root=t1, method="w")
t2 = "(((((0.0056 * ((x1 ** 0.4) * (x2 ** 0.6))) + (0.0068 * x2)) + (0.0091 * x1)) - (0.0806 * y1)) + 0.3831)"
t3 = "(((((0.0056 * ((x1 ** 0.4) * (x2 ** 0.6))) + (684.33 * x2)) + (3.7776 * x1)) - (3.002 * y1)) + 0.0038)"
#%%
temp = "(((((0.969 * ((x1 ** 0.3) * (x2 ** 0.4))) + (0.08 * x2)) + (0.002 * x1)) - (0.944 * y1)) + 0.012)"
x1 = [1 ,1 ,1]
x2 = [2 ,2 ,2]
y1 = [1 ,1 ,3]
#%%
t = pd.DataFrame(data=[x1, x2, y1])
t.head()
#%%
import os

outname = 'name.csv'

outdir = './dir'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)    

t.to_csv(fullname)
#%%
print(fullname)
#%%
var = {"x1": np.array(x1), "x2":np.array(x2), "y1":np.array(y1)}
ne.evaluate(temp, local_dict=var)
#%%
ne.evaluate(temp, local_dict=t)

# print(pd.eval(temp))
#%%

temp = "(((((0.969 * ((x1 ** 0.3) * (x2 ** 0.4))) + (0.08 * x2)) + (0.002 * x1)) - (0.944 * y1)) + 0.012)"
coes = transform.clean_prog(temp)
print("coes", coes)
ridge = np.sum(np.power(coes, 2))
print("ridge", ridge)
lasso = np.sum(np.abs(coes))
print("lasso", lasso)
#%%
tree.TEMP
#%%
tree.set_TEMP(5)

print(tree.TEMP)
#%%

col_name = ['a', 'b', 'c']
df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [0.7, 0.8, 0.9], [0.2, 0.3, 1.1]]),
                  columns=col_name)
#%%
tree.set_global_DATA(df=df)
# df.columns.to_list()
print(tree.GLOBAL.col_names)
print(tree.GLOBAL.df)
#%%
coes = tree.tree(col_name)
coes.program_print()
t2 = tree.tree(col_name)
t2.program_print()


#%%
t12, t22 = tree.do_xover(coes, t2, version=1.1)
#%%
t12.program_print()
t22.program_print()

#%%
def two():
    return "aa"#, "ss"
temp = []
aa, ss, zz = two()
temp.append(aa)
temp.append(ss)
print(temp)

#%%
class Vertify:
    def __init__(self):
        self.value = 1

x1 = Vertify()
x2 = Vertify()

#%%
if x1 == x2:
    print("yes")
else:
    print("no")
#%%
print(x1)
print(x2)
#%%
col_name = ['a', 'b', 'c']
temp = tree.tree(variables=col_name)
#%%
temp.program_print()
#%%
temp.left.program_print()
temp.left.parent.program_print()
#%%
lchild = temp.left
while lchild:
    print(lchild)
    lchild.parent.program_print()
    lchild.program_print()
    print("--")
    if lchild.left is None:
        lchild_last = lchild
    lchild = lchild.left

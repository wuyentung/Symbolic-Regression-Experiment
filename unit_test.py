#%%
from copy import deepcopy
import random
import pandas as pd
import Experiment.tree as tree
import numpy as np
#%%

col_name = ['x1', 'x2', 'y1']
df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [0.7, 0.8, 0.9], [0.2, 0.3, 1.1]]),
                  columns=col_name)
#%%
tree.set_global_DATA(df=df)
# df.columns.to_list()
print(tree.GLOBAL.col_names)
print(tree.GLOBAL.df)
#%%
t1 = tree.tree(col_name)
t1.program_print()
t2 = tree.tree(col_name)
t2.program_print()


#%%
t1.program_print()
t1_vars = t1.get_var_leaves
for var in t1_vars:
    print(var.value)

#%%
t1_vars[0].parent.program_print()


#%%
def two():
    return "aa"#, "ss"

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
import Experiment.tree as tree
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
#%%
parent = lchild_last.parent
while parent:
    print(parent)
    parent.program_print()
    print("--")
    if parent.parent is None:
        parent_last = parent
    parent = parent.parent
#%%
parent_last.program_print()
#%%
tree.GLOBAL.col_names
#%%
tree.set_global_DATA(col_name)
#%%

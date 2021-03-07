#%%
import tree
import pandas as pd
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
r1 = tree.tree(col_name)
r1.program_print()
r2 = tree.tree(col_name)
r2.program_print()
#%%
r12, r22 = tree.do_xover(r1, r2, version=3)

#%%
r12.program_print()
r22.program_print()


#%%
temp = [0, 
1, 
2, 3]
#%%
temp2 = [0, 
# 1, 
2, 3]
#%%

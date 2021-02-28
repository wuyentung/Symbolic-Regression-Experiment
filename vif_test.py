#%%
import data_generate_process
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.formula.api import ols

#%%
DATA, Y_TRUE = data_generate_process.dgp(method="MIMO_1", n=500)
#%%
DATA["x1x2"] = DATA["x1"]*DATA["x2"]
DATA["c"] = 1
#%%
DATA = DATA.drop(columns="y1")
#%%
DATA = DATA.drop(columns="x1")
#%%
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = DATA.columns 

# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(DATA.values, i) 
						for i in range(len(DATA.columns))] 

print(vif_data)

# %%

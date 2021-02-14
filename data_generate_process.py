#%%
from copy import error
import pandas as pd
import numpy as np
#%%
## data generate

np.random.seed(0)


def dgp(method="MIMO_1", n=500):
    if "MIMO_1" == method:
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
        coe_x1x2 = 1.1
        pow_x1 = 0.3
        pow_x2 = 0.4
        coe_y1_eff = (-1)

        y1_eff = coe_x1x2 * (x.T[0] ** pow_x1) * (x.T[1] ** pow_x2) / (slope + 1)
        y2_eff = coe_x1x2 * (x.T[0] ** pow_x1) * (x.T[1] ** pow_x2) + coe_y1_eff * y1_eff
        # y2_eff =  2 * (x.T[0] * x.T[1]  )# -  y1_eff
        # y2_eff = x.T[0] ** 0.3#* x.T[1]  )# -  y1_eff

        y1_eff = y1_eff.reshape(n, 1)

        ## round
        y2_eff = np.round(y2_eff, 2)
        y1_eff = np.round(y1_eff, 2)

        x = np.round(x, 2)


        ## data for SR
        d = {'y2': y2_eff.T,
            'x1': x.T[0],
            'x2': x.T[1],
            'y1': y1_eff.T[0],
            }
        #%%
        data = pd.DataFrame(data=d)
        data.to_csv("dgp_%s.csv" %method, header=True, index=False)
        # data = pd.read_csv("dgp_%s.csv" %method)
        y_true = data.pop("y2")
        if __name__ == "__main__":
            print(data.head())
            print(data.head())
        # print(data.columns.to_list())
        return data, y_true
    raise SyntaxError("invalid method for dgp")
method = "MIMO_1"
n = 500

# %%

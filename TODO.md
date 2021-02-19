# Production function properties (according to *Cobb-Douglas* production function)
### Q = A * L^α * K^β
### Give a penalty if expression doesn't follow **one** of it or **all**
- Parameters:
    1. A, α and β is positive constant
        1. ~~A > 0:~~
            - ~~bottom up, len(["minus"]+["-" and rightsub]) //2 == 0~~
        1. α, β is consant:
            - ~~bottom up, encounter "**" and leftsub~~
            - and which "**" rightsub is Numbers (len(get_leaves) == 0)
            1. α, β > 0: (top-down)
                - ~~"**" rightsub > 0~~
            1. α, β ≤ 1:
                - sum("**" rightsub) ≤ 1
    1. we define α + β ≤ 1
- For **INPUTS**
    1. **positive effect** for each input
    1. multipling relationship for each other
    1. sum of the expoential ≤ 1 
- For **OUTPUTS**
    1. **trade-off effect** for each output

https://www.economicsdiscussion.net/production-function/cobb-douglas-production-function-and-its-properties/23407

# Production Function Properties (Hard Constraint)
## use minus trade-off for output first
1. [x] Function: coe = num_tree(var=None), well it should be built in scientific expression
1. [x] Function: coe * var ^ coe, s.t. coe > 0, for x: ≥ 0
1. [x] structure of Cobb-Douglas
1. [x] short express for coe
1. [x] crossover: Fix point crossover for certain node

# Transform Result
1. [x] 把所有係數弄下來
1. [x] cal mean
1. [x] cal var

# EN in fitness
1. [x] RSS
1. [_] coes of all(import transform_result)
1. [_] ridge
1. [_] lasso


1. 重複 30 次跑回歸係數
如果是跑出
係數估計用統計的方法
比較與 SR 的優劣在小問題，生產函數相對簡單合理
如果是複雜的問題就只能用 SR

    https://scipy-cookbook.readthedocs.io/items/robust_regression.html
    https://hernandis.me/2020/04/05/three-examples-of-nonlinear-least-squares-fitting-in-python-with-scipy.html

    https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    - 用 scipy 跑 closed－form 的 nonlinear regression

1. 在用 EN 的過程發現有共線性，所以用 scad
1. ocba 縮小後幾次實驗的樹的變數選擇（必須要在環境單純下使用

## 02/12/2021
1. [x] auto-make experiment directory
1. [x] ne.evaluate((02/13

## 02/14/2021
1. [x] refactor DGP to module

## 02/15/2021
1. [x] refactor Record to module
1. [_] non-linear regression

# effectiveness
## high lasso rate -> hi bias, lo var
1. (fitness *20 30 no change then stop GP) * 30 exp
1. maybe fitness function is too simple, 貢獻性（vif）低
## to get low bias
1. SCAD or MCP

## 把流程圖加上各種方法的情況

## 實驗
1. 對四個方法，兩個、五個、十個變數的時間、fitness 比較
1. 
1. 
1. 

### 3 input produciton funtion, 1 output
andy 

## 02/18/2021
1. [x] converge critiria
    1. [x] 20 times fitness varience < 0.01
1. [x] refactor Record 
1. [_] SCAD (v6)
1. [x] run exp_v5_02

## 02/19/2021
1. [x] refactor fig size
1. [x] SCAD (v6.1)
1. [x] MCP (v6.2)
    1. [x] compute_fitness
    1. [x] GLOBAL
    1. [x] set_global
    1. [x] experiment
1. [x] run experiment v5 rest

## 02/20/2021
1. [_] 佑鑫書
1. [_] SCAP, MCP hyperparameter adjust
1. [_] 課程錄影

## before next meeting
1. [_] flow_chart for my program
    1. [_] incidence and responce
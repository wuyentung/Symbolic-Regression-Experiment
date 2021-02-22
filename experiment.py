#%%
import tree
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import transform_result as transform
import os
import data_generate_process
from record import Record
#%%
# get data
DATA, Y_TRUE = data_generate_process.dgp(method="MIMO_1", n=500)

#%%
### main
# noinspection PyTypeChecker
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

    if exp_name == "eriment":
        check = True
        MAX_GENERATIONS = 25
        EXP_TIMES = 5
        POP_SIZE = 5
    tree.set_global_DATA(df=DATA)
    tree.set_global_POP_SIZE(POP_SIZE=POP_SIZE)

    if EN_ridge_ratio:
        tree.set_global_EN_ridge_ratio(EN_ridge_ratio=EN_ridge_ratio)
        print("use EN_ridge_ratio: %f" %tree.GLOBAL.EN_ridge_ratio)

    if EN_lamda:
        tree.set_global_EN_lamda(EN_lamda=EN_lamda)
        print("use EN_lambda: %f" %tree.GLOBAL.EN_lamda)

    if SCAD_a:
        FITNESS_METHOD = "SCAD"
        tree.set_global_SCAD_a(SCAD_a=SCAD_a)
        print("use SCAD_a: %f" %tree.GLOBAL.SCAD_a)

    if SCAD_lamda:
        tree.set_global_SCAD_lamda(SCAD_lamda=SCAD_lamda)
        print("use SCAD_lambda: %f" %tree.GLOBAL.SCAD_lamda)

    if MCP_a:
        FITNESS_METHOD = "MCP"
        tree.set_global_MCP_a(MCP_a=MCP_a)
        print("use MCP_a: %f" %tree.GLOBAL.MCP_a)

    if MCP_lamda:
        tree.set_global_MCP_lamda(MCP_lamda=MCP_lamda)
        print("use MCP_lambda: %f" %tree.GLOBAL.MCP_lamda)

    ## experiment starts here
    for i in range(EXP_TIMES):
        print()
        print("-------------------")
        print("%d time experiment" %i)
        print("-------------------")
        print()
        # NEW_POP_PCT = 0.1
        col_name = DATA.columns.to_list()
        population = [tree.tree(col_name) for _ in range(POP_SIZE)]
        TOURNAMENT_SIZE = 3
        XOVER_PCT = 0.7
        REG_STRENGTH = 2
        select_depth = 10


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
            leaf_counts = []
            best_prog = population[0]
            for prog in population:
                # print(kk)
                kk += 1
                # prog.program_print()
                leaf_counts.append(prog.leaf_count)
                prediction = tree.evaluate(prog, df=DATA)
                # print(type(prediction))
                # print(prediction)
                score = tree.compute_fitness(prog, prediction, REG_STRENGTH, Y_TRUE, method=FITNESS_METHOD)
                # print(score)
                if np.isnan(score):
                    score = np.inf
                fitness.append(score)
                if np.isinf(score):
                    continue

                if score < global_best:
                    global_best = score
                    best_prog = prog

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
                offspring1, offspring2 = tree.get_offspring(population, fitness, POP_SIZE, col_name,TOURNAMENT_SIZE, XOVER_PCT, version=1.1)
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
                tree.lineNotifyMessage(token, message)

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
    plt.xlabel("generation")  # x label
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

        tree.lineNotifyMessage(token, message)

    return exp_Records

    
#%%
do = False
if do:
    exp_V1_12 = experiment(exp_name="V1_12")
    exp_V1_2 = experiment(exp_name="V1_2")
do_v122 = False
if do_v122:
    exp_V1_22 = experiment(exp_name="V1_22")
do_v2 = False
if do_v2:
    exp_V1 = experiment(exp_name="V2")
do_v21 = False
if do_v21:
    exp_v21 = experiment(exp_name="V21")
    exp_V1 = experiment(exp_name="V2")
do_v3 = False
if do_v3:
    name = "V3"
    exp_v3 = experiment(exp_name=name)
    transform.coe_substract(name)
do_v31 = False
if do_v31:
    name = "V31"
    exp_v31 = experiment(exp_name=name)
    transform.coe_substract(name)
#%%
do_v32 = False
do_v32_06 = do_v32
if do_v32_06:
    name = "V32_06"
    exp_v32_06 = experiment(exp_name=name, EN_ridge_ratio=0.6)
    transform.coe_substract(name)
do_v32_07 = do_v32
if do_v32_07:
    name = "V32_07"
    exp_v32_07 = experiment(exp_name=name, EN_ridge_ratio=0.7)
    transform.coe_substract(name)
do_v32_08 = do_v32
if do_v32_08:
    name = "V32_08"
    exp_v32_08 = experiment(exp_name=name, EN_ridge_ratio=0.8)
    transform.coe_substract(name)
#%%
do_v5_02 = False
# _ridgeRatio_Lamda
if do_v5_02:
    EN_ridge_ratio = 0.2
    name = "V5_02_2"
    exp_v5_02_2 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=2)
#%%
if do_v5_02:
    EN_ridge_ratio = 0.2
    name = "V5_02_5"
    exp_v5_02_5 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=5)

    name = "V5_02_10"
    exp_v5_02_10 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=10)
#%%
do_v5_05 = True
# _ridgeRatio_Lamda
if do_v5_05:
    EN_ridge_ratio = 0.5
    
    name = "V5_05_2"
    exp_v5_05_2 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=2)

    name = "V5_05_5"
    exp_v5_05_5 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=5)

    name = "V5_05_10"
    exp_v5_05_10 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=10)
#%%
do_v5_08 = False
# _ridgeRatio_Lamda
if do_v5_08:
    EN_ridge_ratio = 0.8

    name = "V5_08_2"
    exp_v5_08_2 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=2)

    name = "V5_08_5"
    exp_v5_08_5 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=5)

    name = "V5_08_10"
    exp_v5_08_10 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=10)
#%%
do_v601 = False
# _ridgeRatio_Lamda
if do_v5_08:
    EN_ridge_ratio = 0.8

    name = "V5_08_2"
    exp_v5_08_2 = experiment(exp_name=name, EN_ridge_ratio=EN_ridge_ratio, EN_lamda=2)

#%%
expTemp = experiment()
#%%
# %%

#%%
import Experiment.tree as tree
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
#%%
class Record(object):
    import matplotlib.pyplot as plt
    def __init__(self):
        self.generation = []
        self.fitness = []
        self.G_count = 0
        self.avgs_leaf_count = []
        self.bests_leaf_count = []
        self.best_programs = []
        self.time_used = []

    def _update_time_used(self, t):
        self.time_used.append(t)
    def _update_fitness(self, value):
        '''

        :param value: fitness value of this generation
        :type value: float
        :return:
        '''
        self.fitness.append(value)
        return None

    def _update_leaf_counts(self, leaf_counts, best_count):
        '''
        :param leaf_counts: list of leaf counts in generation
        :type leaf_counts: list
        :param best_count:
        :type best_count: int
        :return:
        '''
        avg = round(np.average(leaf_counts), 1)
        self.avgs_leaf_count.append(avg)
        self.bests_leaf_count.append(best_count)

    def _update_program(self, program):
        '''

        :param program:
        :return:
        '''
        self.best_programs.append(program)

    def update_all(self, fitness=False, leaf_counts=False, best_count=False, program=False, t=False, not_final=True):
        '''

        :param t: by sec
        :param fitness:
        :param leaf_counts:
        :param best_count:
        :param program:
        :return:
        '''
        if not_final:
            self.generation.append(self.G_count)
            self.G_count +=1
            self._update_fitness(fitness)
            self._update_leaf_counts(leaf_counts, best_count)
        if program:
            self._update_program(program)
            self._update_time_used(t)

    def _show_fitness(self, save=False):
        plt.plot(self.generation, self.fitness)
        if save:
            plt.savefig('%s.png' %save , dpi=600, format='png')
        plt.show()

    def _show_leaves(self, save=False):
        plt.plot(self.generation, self.avgs_leaf_count, 'r--', self.generation, self.bests_leaf_count, 'bs')
        if save:
            plt.savefig('%s.png' %save , dpi=600, format='png')
        plt.show()

    def _save_program_n_time(self, file_name):
        file = pd.DataFrame(data=[self.best_programs, self.time_used], index=["best_programs", "time_used"]).T
        file.to_csv("%s.txt" % file_name , index=False)
    def save_all(self, exp_id):
        exp_fitness = "exp" + str(exp_id) + "_fitness"
        exp_leaves = "exp" + str(exp_id) + "_leaves"
        exp_program = "exp" + str(exp_id) + "_program"
        self._show_fitness(save=exp_fitness)
        self._show_leaves(save=exp_leaves)
        self._save_program_n_time(exp_program)
        return "Files saved"

#%%
## data generate

np.random.seed(0)

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


# %%

## data for SR
d = {'y2': y2_eff.T,
     'x_1': x.T[0],
     'x_2': x.T[1],
     'y_1': y1_eff.T[0],
     }

data = pd.DataFrame(data=d)
print(data.head())
y_true = data.pop("y2")
print(data.head())
print(data.columns.to_list())
# %%

### main
# noinspection PyTypeChecker
def experiment(exp_name = "eriment"):
    recording = []
    MAX_GENERATIONS = 100
    EXP_TIMES = 50
    POP_SIZE = 500

    if exp_name == "eriment":
        MAX_GENERATIONS = 20
        EXP_TIMES = 5
        POP_SIZE = 200
    for i in range(EXP_TIMES):
        # NEW_POP_PCT = 0.1
        col_name = data.columns.to_list()
        population = [tree.tree(col_name) for _ in range(POP_SIZE)]
        ## TODO: revise old program
        TOURNAMENT_SIZE = 3
        XOVER_PCT = 0.7
        REG_STRENGTH = 2
        select_depth = 10


        global_best = float("inf")
        unchanged_score = 0
        ts = time.time()
        recording.append(Record())
        for gen in range(MAX_GENERATIONS):
            t1 = time.time()
            fitness = []
            change_flag = 1
            kk=0
            leaf_counts = []
            for prog in population:
                # print(kk)
                kk += 1
                # prog.program_print()
                leaf_counts.append(prog.leaf_count)
                prediction = [tree.evaluate(prog, data)]
                # print(type(prediction))
                score = tree.compute_fitness(prog, prediction, REG_STRENGTH, y_true)
                # print(score)
                if np.isnan(score):
                    score = np.inf
                fitness.append(score)
                if np.isinf(score):
                    continue

                if score < global_best:
                    global_best = score
                    best_pred = prediction
                    best_prog = prog
                    change_flag = 0
            if change_flag:
                unchanged_score = unchanged_score + 1
            else:
                unchanged_score = 0
            # print(unchange_score)

            # noinspection PyUnresolvedReferences
            prog_express = best_prog.program_express
            t2 = time.time()


            ## recording
            # noinspection PyTypeChecker
            recording[i].update_all(fitness=global_best, leaf_counts=leaf_counts, best_count=best_prog.leaf_count,program=prog_express, t=(t2-t1))
            print(
                "\nunchange_score: %d\nGeneration: %d\nBest Score: %.5f\nMedian score: %.5f\nBest program: %s\nTime used: %d sec\n"
                % (
                    unchanged_score,
                    gen,
                    global_best,
                    pd.Series(fitness).median(),
                    prog_express,
                    t2 - t1,
                )
            )

            # best_count = best_prog.size
            NEW_POP_PCT = 0.05
            next_population = []
            if exp_name == "eriment":
                next_population_incomplete = [
                    tree.get_offspring(population, fitness, POP_SIZE, col_name, TOURNAMENT_SIZE, XOVER_PCT)
                    for _ in range(round(POP_SIZE * (1.0 - NEW_POP_PCT)) - 1)]
                next_population_incomplete.append(best_prog)
                # print("new gen")

                new_pop = [tree.tree(col_name) for _ in range(round(POP_SIZE * NEW_POP_PCT))]
                next_population = next_population_incomplete + new_pop
            else:
                for _ in range(int(POP_SIZE/2)):
                    # print(type(population[0]))
                    # print(type(fitness[0]))
                    # print(type(POP_SIZE))
                    # print(type(col_name[0]))
                    offspring1, offspring2 = tree.get_offspring(population, fitness, POP_SIZE, col_name, TOURNAMENT_SIZE, XOVER_PCT, version=1.1)
                    next_population.append(offspring1)
                    next_population.append(offspring2)
                next_population[0] = best_prog

            population = next_population

        tf = time.time()
        print("Best score: %f" % global_best)
        print("Best program: %s" % prog_express)
        print("Total time: %d sec" % (tf - ts))


    # 修改為你要傳送的訊息內容
    m = "\nTotal time: %d sec" % (tf - ts)
    message = "\n\nexp_" + str(exp_name) + "complete" + m
    # 修改為你的權杖內容
    token = 'CCgjmKSEGamkEj9JvhuIkFNYTrpPKHyCb1zdsYRjo86'

    tree.lineNotifyMessage(token, message)

    Final_record = Record()
    for i in range(EXP_TIMES):
        Final_record.update_all(program=recording[i].best_programs[-1], t=round(np.sum(recording[i].time_used), 2), not_final=False)

    for g in range(MAX_GENERATIONS):
        fitnesses = []
        leaf_counts = []
        best_leaf_counts = []
        for i in range(EXP_TIMES):
            # print(i, g)
            fitnesses.append(recording[i].fitness[g])
            leaf_counts.append(recording[i].avgs_leaf_count[g])
            best_leaf_counts.append(recording[i].bests_leaf_count[g])

        Final_record.update_all(np.average(fitnesses), leaf_counts, np.average(best_leaf_counts))
    Final_record.save_all(exp_id=exp_name)
    return Final_record
#%%
exp_temp = experiment(exp_name="V1.1")

# print(Final_record.best_programs)
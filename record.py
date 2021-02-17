#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#%%
class Record(object):
    '''
    Final_record -> 
    :type fitness: [[]]
    :type avgs_leaf_count: [[]]
    '''
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

    def _show_mean_of_fitness(self, file_name):
        mean_of_fitness = []
        for i in range(self.G_count):
            mean_of_fitness.append(np.mean(self.fitness[i]))
        plt.plot(self.generation, mean_of_fitness)
        plt.title("mean_of_fitness change")  # title
        plt.ylabel("mean_of_fitness")  # y label
        plt.xlabel("generation")  # x label
        plt.savefig('%s.png' % file_name, dpi=600, format='png')
        plt.show()

    def _show_var_of_fitness(self, file_name):
        var_of_fitness = []
        for i in range(self.G_count):
            var_of_fitness.append(np.var(self.fitness[i]))
        plt.plot(self.generation, var_of_fitness)
        plt.title("var_of_fitness change")  # title
        plt.ylabel("var_of_fitness")  # y label
        plt.xlabel("generation")  # x label
        plt.savefig('%s.png' % file_name, dpi=600, format='png')
        plt.show()

    def _show_leaves(self, file_name):
        plt.plot(self.generation, self.avgs_leaf_count, 'r--', self.generation, self.bests_leaf_count, 'bs')
        plt.title("leaves change")  # title
        plt.ylabel("n_leaf")  # y label
        plt.xlabel("generation")  # x label
        plt.savefig('%s.png' % file_name, dpi=600, format='png')
        plt.show()

    def _save_program_n_time(self, file_name):
        file = pd.DataFrame(data=[self.best_programs, self.time_used], index=["best_programs", "time_used"]).T
        file.to_csv("%s.txt" % file_name , index=False)
    def save_all(self, exp_id):

        # making directicory if doesn't excist 
        out_dir = "./%s" %(exp_id)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        # naming experiment stuff
        exp_mean_of_fitness = str(exp_id) + "_mean_of_fitness"
        exp_var_of_fitness = str(exp_id) + "_var_of_fitness"
        exp_leaves = str(exp_id) + "_leaves"
        exp_program = str(exp_id) + "_program"

        self._show_mean_of_fitness(file_name=os.path.join(out_dir, exp_mean_of_fitness))
        self._show_var_of_fitness(file_name=os.path.join(out_dir, exp_var_of_fitness))
        # self._show_leaves(file_name=os.path.join(out_dir, exp_leaves))
        self._save_program_n_time(file_name=os.path.join(out_dir, exp_program))
        return "Files saved"

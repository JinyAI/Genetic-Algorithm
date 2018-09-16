# -*- coding: utf-8 -*-
# Copyright 2018, JinYoung Kim Softcomputing LAB all rights reserved.

"""
Created on Wed Sep 12 16:22:27 2018

@author: ys1
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

def str2mem(string):
    idx_str = ''
    for i in range(len(string)):
        if string[i] == 'C':
            idx_str += '0'
        elif string[i] =='D':
            idx_str += '1'
        else:
            print("Error: chromosome has to consist of 'C' or 'D'")
    return int(idx_str, base=2)

def mem2str(mem):
    results = ''
    bin_idx = bin(mem)[2:]
    while len(bin_idx) < 2**(length*2):
        bin_idx = '0' + bin_idx
    for c in bin_idx:
        if c =='0':
            results +='C'
        elif c=='1':
            results += 'D'
    return results

def make_dict(length):
    pop_dict = {}
    for i in range(2**2**(length*2)):
        pop_dict[mem2str(i)] = 0
    return pop_dict

def roulette_wheel(individuals):
    fitnesses = np.array([ind.fitness for ind in individuals])
    roulette = fitnesses / np.sum(fitnesses)
    ind = np.random.choice(individuals, p=roulette)
    return Individual(ind.length, chromosome=ind.chromosome)

def custom(other, i, trigger_flag, history):
    if other == 'Trigger':
        if trigger_flag:
            return 'D'
        else:
            return 'C'
    elif other == 'AllD':
        return 'D'
    elif other == 'CDCD':
        if i%2==0:
            return 'C'
        else:
            return 'D'
    elif other == 'CCD':
        if i%3==0:
            return 'C'
        elif i%3==1:
            return 'C'
        else:
            return 'D'
    elif other =='Random':
        if np.random.randint(2)==1:
            return 'D'
        else:
            return 'C'
    elif other =='TFT':
        if history =='C':
            return 'C'
        elif history =='D':
            return 'D'


class Individual:
    def __init__(self, length, chromosome = 'default', random = False):
        self.length = length
        self.fitness = 0
        self.FIRST_ACTION = - 2 * length
        if chromosome == 'default':
            self.chromosome = self.generate_chromosome(self.length, random)
        else:
            self.chromosome = list(chromosome)
            
    def generate_chromosome(self, length, random):
        chromosome = []
        #for _ in range(2**(length * 2) + 2 * length):
        #    if np.random.randint(2) ==1:
        #        chromosome.append('C')
        #    else:
        #        chromosome.append('D')
        #return chromosome
        if random:
            for _ in range(2**(length * 2) + 2 * length):
                if np.random.randint(2) ==1:
                    chromosome.append('C')
                else:
                    chromosome.append('D')
            return chromosome
        else:
            for _ in range(2**(length * 2) + 2 * length):
                if np.random.randint(2) ==1:
                    chromosome.append('D')
                else:
                    chromosome.append('D')
            return chromosome
        
    def mutate(self, n = 1):
        for i in range(n):
            x = np.random.randint(0, len(self.chromosome))
            if self.chromosome[x] == "D":
                self.chromosome[x] = "C"
            else:
                self.chromosome[x] = "D"
        return Individual(self.length, chromosome=self.chromosome)
                
    def mate(self, other):
        new_ind = Individual(self.length, self.chromosome)
        new_ind2 = Individual(self.length, other.chromosome)
        idx = np.random.randint(0, len(new_ind.chromosome))
        #length = np.random.randint(0, len(new_ind.chromosome) - idx)
        for i in range(idx):
            new_ind2.chromosome[i] = self.chromosome[i]
        for i in range(idx, len(self.chromosome)):
            new_ind.chromosome[i] = other.chromosome[i]
        return [new_ind, new_ind2]

    def get_next_move(self, memory_str):
        memory = str2mem(memory_str)
        return self.chromosome[memory]
    
    def fight(self, other, string = False, iterations = 40):
        score_mine = 0
        score_other = 0
        trigger_flag = 0
        memory_self_str = self.chromosome[self.FIRST_ACTION:]
        history = 'C'
        if not string:
            memory_other_str = other.chromosome[other.FIRST_ACTION:]
        for i in range(iterations):
            mymove = self.get_next_move(memory_self_str)
            if not string:
                enmove = other.get_next_move(memory_other_str)
            else:
                enmove = custom(other, i, trigger_flag, history)
            # evaluate scores
            if mymove == "C" and enmove == "C":
                score_mine += 3
                score_other += 3
                history = 'C'
            if mymove == "C" and enmove == "D":
                score_mine += 0
                score_other += 5
                history = 'C'
            if mymove == "D" and enmove == "C":
                trigger_flag = 1
                score_mine += 5
                score_other += 0
                history = 'D'
            if mymove == "D" and enmove == "D":
                trigger_flag = 1
                score_mine += 1
                score_other += 1
                history = 'D'
            memory_self_str = memory_self_str[2:] + [mymove] + [enmove]
            if not string:
                memory_other_str = memory_other_str[2:] + [enmove] + [mymove]
            
        return (score_mine/iterations, score_other/iterations)
    
    def test(self, fixed_ind, string = False):
        for ind in fixed_ind:
            scores = self.fight(ind, string)
            self.fitness += scores[0]
        self.fitness /= len(fixed_ind)

class Population:
    def __init__(self, n_pop, length, n_best, mutate_rate, mate_rate):
        self.n_pop = n_pop
        self.individuals = []
        self.length = length
        self.n_best = n_best
        self.mutate_rate = mutate_rate
        self.mate_rate = mate_rate
        if n_best > self.n_pop // 2:
            print("Error: the number of selected individuals must be lower than half of n_pop")
            return -1
        
    def size(self):
        return len(self.individuals)
        
    def create_generation(self, random = False):
        for _ in range(self.n_pop):
            self.individuals.append(Individual(self.length, random=random))
    
    def eval_fitness(self, debug=False):
        for ind in self.individuals:
            ind.fitness = 0
            
        for ind in self.individuals:
            ind.test(fixed_ind, string=True)
        
        #for i in range(len(self.individuals)):
        #    for j in range(len(self.individuals)):
        #        if i==j:
        #            continue
        #        i1 = self.individuals[i]
        #        i2 = self.individuals[j]
        #        scores = i1.fight(i2)
        #        i1.fitness += scores[0]
    

    
    def print_fitness(self):
        m = -1
        M = -1
        aver = 0
        pop_dict = make_dict(self.length)
        self.eval_fitness()
        for ind in self.individuals:
            if m==-1:
                m, M = ind.fitness, ind.fitness
            if m > ind.fitness:
                m = ind.fitness
            if M < ind.fitness:
                M = ind.fitness
            aver += ind.fitness
            pop_dict[''.join(ind.chromosome)[:-2*self.length]] += 1
        return (M, m, aver/len(self.individuals), pop_dict)
    
    def select(self):
        self.eval_fitness()
        self.individuals = sorted(self.individuals, key=lambda a: a.fitness, reverse = True)
        
        #self.individuals = sorted(self.individuals, key=lambda a: a.fitness, reverse = True)
        
        #best_ind = new_ind[:self.n_best]
        #worst_ind = new_ind[self.n_best:]
        #self.individuals = best_ind
        #for _ in range(self.n_best):
        #    self.individuals.append(worst_ind[np.random.randint(len(worst_ind))])
    
    def next_gen(self):
        new_pop = Population(self.n_pop, self.length, self.n_best, self.mutate_rate, self.mate_rate)
        self.select()
        
        #new_pop.individuals = list(np.random.choice(self.individuals, p=self.roulette, size=[self.n_best]))
        
        for i in range(self.n_best):
            new_pop.individuals.append(Individual(self.length, self.individuals[i].chromosome))
        
        #for ind in self.individuals[self.n_best:self.n_best*2]:
        #    new_pop.individuals.append(ind.mutate(1 if np.random.randint(100) <= self.mutate_rate*100 else 0))        
        while new_pop.size() < self.n_pop:
            ind = roulette_wheel(self.individuals)
            ind2 = roulette_wheel(self.individuals)
            offsprings = [ind, ind2]
            if np.random.randint(100) <= self.mate_rate * 100:
                offsprings = ind.mate(ind2)
            for j in range(len(offsprings)):
                if np.random.randint(100) <= self.mutate_rate * 100:
                    new_pop.individuals.append(offsprings[j].mutate(1))
                else:
                    new_pop.individuals.append(offsprings[j])            
        return new_pop
    
class PrisonerDilemma_GA:
    def run(n_pop = 100, length = 1, n_best = 10, mutate_rate = 0.2, mate_rate=0.2, generations = 50):
        pop = Population(n_pop, length, n_best, mutate_rate, mate_rate)
        pop.create_generation(random=True)
        M_list, m_list, aver_list, pop_dict_list = [], [], [], []
        M, m, aver, pop_dict = pop.print_fitness()
        M_list.append(M)
        m_list.append(m)
        aver_list.append(aver)
        pop_dict_list.append(pop_dict)
        print(datetime.datetime.now(), "%03d th generations... Max: %f, Min: %f, Average: %f"%(1, M, m, aver))
        for i in range(1, generations):
            pop = pop.next_gen()
            M, m, aver, pop_dict = pop.print_fitness()
            M_list.append(M)
            m_list.append(m)
            aver_list.append(aver)
            pop_dict_list.append(pop_dict)
            print(datetime.datetime.now(), "%03d th generations... Max: %f, Min: %f, Average: %f"%(i+1, M, m, aver))
        return pop, [M_list, m_list, aver_list], pop_dict_list
    
    
n_pop = 20
length = 1

#fixed_ind = [
#        Individual(length, [c for c in 'CCCCCC']),
#        Individual(length, [c for c in 'DDDDDD']),
#        Individual(length, [c for c in 'CDCDCC']),
#        Individual(length, [c for c in 'CDCDCD']),
#        Individual(length, [c for c in 'DDCCDD']),
#        Individual(length, [c for c in 'DDCCCC'])
#        ]

# For AllC test
#fixed_ind = [
#        Individual(length, [c for c in 'CCCCCC'])
#        ]

# For random test
#pop = Population(10, length, n_best = 1, mutate_rate = 0.2, mate_rate=0.2)
#pop.create_generation(random=True)
#fixed_ind = pop.individuals
#fixed_ind_chr = [''.join(ind.chromosome) for ind in fixed_ind]

# For below test. If you use below fixed individual, you mush set string in fight method of Individual class as TRUE, if not FALSE
fixed_ind = ['Trigger','AllD','CDCD','CCD','Random', 'TFT']

pop, lists, pop_dict = PrisonerDilemma_GA.run(n_pop = n_pop, length = length, n_best = 4, mutate_rate = 0.01, 
                             mate_rate = 0.5, generations = 50)


plt.figure(figsize=(8,8))
for i in range(3):
    plt.plot(np.array(lists[i]))
plt.legend(['M', 'm', 'aver'])
plt.show()

legend = []
plt.figure(figsize=(6.5,8))
for i in range(2**2**(length*2)):
    code = mem2str(i)
    tmp = []
    for k in range(len(pop_dict)):
        tmp.append(pop_dict[k][code])
    plt.plot(np.array(tmp))
    legend.append(code)
plt.legend(legend,loc=2, bbox_to_anchor=(1.05,0.8))
plt.show()

chromosomes = [''.join(ind.chromosome) for ind in pop.individuals]
for i,i1 in enumerate(pop.individuals):
    for j,i2 in enumerate(pop.individuals):
        if i==j:
            continue
        if i1==i2:
            print(i,j,'duple error')
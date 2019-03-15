#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#####################################################
#
#
#       evo-devo of polyphenisms.py
#
#       Author: Roman Zug (roman.zug@biol.lu.se)
#
#
#####################################################
"""

# IMPORTING

import numpy as np
from scipy import optimize
#import matplotlib.pyplot as plt
#from tqdm import tqdm
import progressbar


# PARAMETERS
print()

# population size
n = 500
print("Population size:", n)

# total number of generations
no_gens = 2000
print("Generations:", no_gens)

print()

# scaling constant for deterministic factors (when computing E)
A = 1
print("A:", A)

# scaling constant for stochastic factors (when computing E)
B = 0
print("B:", B)

# number of time steps per generation (= life span) (when computing E)
L = 1
print("L:", L)

# relative timescale of environmental variation (number of generations per environmental cycle) (when computing E)
# R small (R=1): fast environmental change. R large: slow environmental change
R = 500
print("log R:", np.log10(R)) 

# environmental predictability
P = 0.9
print("P:", P)

# constant that determines strength of fitness decay due to phenotypic mismatch
tau = 1
print("tau:", tau)

# constant that determines strength of fitness gain due to advantage of rare/extreme phenotypes
phi = 4
print("phi:", phi)

# cost for developmental plasticity
k_d = 0.01
print("k_d:", k_d)

# mutation rate
mu = 0.001
print("mu:", mu)

print()


# FUNCTION DEFINITIONS   


def environment(t):
    """sets environmental variation and calculates corresponding cue"""
    
    # E: environmental condition (e.g. temperature)
    eps = 1 - np.random.random() # stochastic error term
    E = A * (np.sin(2 * np.pi * t / (L * R)) + 1) / 2 + B * eps
    
    # C: environmental cue (e.g. photoperiod)
    mu_env = 0.5 * (1 - P) + P * E
    sigma_env = 0.5 * (1 - P) / 3
    C = np.random.normal(mu_env, sigma_env)
    if C <= 0.0:
        C = 1e-09
    if C <= 0.0:
        raise ValueError("C is lower than or equal to zero.")
    return E, C


def create_population(n):
    """creates the starting population of n individuals"""
    
    # E: environmental condition
    # C: environmental cue
    # G: genetic component of basal expression of x
    # b: degree to which basal expression of x depends on environment
    # basal: basal expression of the lineage factor x (basal = G + b*C)
    # alpha: strength of the feedback (in the ODE); initial value: nicht mehr 1e-09, sondern 1
    # hill: Hill coefficient (in the ODE); initial value: 1
    
    population = np.zeros(n, dtype=[('E','<f8'), ('C','<f8'), ('G','<f8'), ('b', '<f8'), 
                                    ('basal','<f8'), ('alpha','<f8'), ('hill','<f8'), 
                                    ('phenotype','<f8'), ('fitness','<f8')])
    population['G'] = 1 - np.random.random(n)
    population['b'] = 1 - np.random.random(n)
    population['alpha'] = 1
    population['hill'] = 1
    return population


# phenotype function with scipy.optimize.root
def phenotype(population):
    """creates the phenotype"""

    population['basal'] = population['G'] + population['b'] * population['C']

    if np.any(population['basal'] <= 0):
        raise ValueError("population['basal'] is lower than or equal to zero.")

    if np.any(population['alpha'] <= 0):
        raise ValueError("population['alpha'] is lower than or equal to zero before mutation.")

    def fun(x):
        x = np.maximum(1e-8, x)
        return (population['basal'] - x + population['alpha'] 
                * (x ** population['hill'] / (1 + x ** population['hill'])))
    
    sol = optimize.root(fun, np.zeros(n), method='excitingmixing')#, options=dict(fatol=1e-8))
    return sol.x



def fitness(population):
    """calculates the fitness"""
    
    mean_phenotype = np.mean(population['phenotype'])
    # fitness_part_1: fitness component deriving from phenotypic (mis)match
    fitness_part_1 = (np.exp(-np.abs(population['E'] - population['phenotype']) * tau))
    # fitness_part_2: fitness component deriving from disruptive selection (advantage of rare phenotypes)
    fitness_part_2 = (1 - np.exp(-np.abs(mean_phenotype - population['phenotype']) * phi))
    W = (fitness_part_1 * fitness_part_2) - k_d
    W[np.flatnonzero(W < 0.0)] = 0.0
    return W


def mutation(population):
    """introduces random mutations for the genetic parameters"""
    
    n = len(population)
    
    # mutation of G; 1e-09 <= G <= 1
    indices_muta_G, = np.where(np.random.random(n) <= mu) # comma is important!
    number_muta_G = indices_muta_G.size
    population['G'][indices_muta_G] += np.random.normal(0, 0.05, number_muta_G)
    # lower bound: 1e-09
    non_pos_indices_G, = np.where(population['G'] <= 0.0) # comma is important!
    population['G'][non_pos_indices_G] = 1e-09
    # upper bound: 1
    population['G'][np.flatnonzero(population['G'] > 1)] = 1
    if np.any(population['G'] <= 0):
        raise ValueError("population['G'] is lower than or equal to zero.")
        
    # mutation of b; 1e-09 <= b <= 1
    indices_muta_b, = np.where(np.random.random(n) <= mu)
    number_muta_b = indices_muta_b.size
    population['b'][indices_muta_b] += np.random.normal(0, 0.05, number_muta_b)
    # lower bound: 1e-09
    non_pos_indices_b, = np.where(population['b'] <= 0.0)
    population['b'][non_pos_indices_b] = 1e-09
    # upper bound: 1
    population['b'][np.flatnonzero(population['b'] > 1)] = 1
    if np.any(population['b'] <= 0):
        raise ValueError("population['b'] is lower than or equal to zero.")
    
    # mutation of alpha; lower bound either 1e-09 or 1
    indices_muta_alpha, = np.where(np.random.random(n) <= mu)
    number_muta_alpha = indices_muta_alpha.size
    population['alpha'][indices_muta_alpha] += np.random.normal(0, 0.05, number_muta_alpha)
    # lower bound: 1e-09
    #non_pos_indices_alpha, = np.where(population['alpha'] <= 0.0)
    #population['alpha'][non_pos_indices_alpha] = 1e-09
    # lower bound: 1
    population['alpha'][np.flatnonzero(population['alpha'] < 1)] = 1
    if np.any(population['alpha'] <= 0):
        raise ValueError("population['alpha'] is lower than or equal to zero after mutation.")
    
    # mutation of hill; lower bound either 1e-09 or 1
    indices_muta_hill, = np.where(np.random.random(n) <= mu)
    number_muta_hill = indices_muta_hill.size
    population['hill'][indices_muta_hill] += np.random.normal(0, 0.05, number_muta_hill)
    # lower bound: 1e-09
    #non_pos_indices_hill, = np.where(population['hill'] <= 0.0)
    #population['hill'][non_pos_indices_hill] = 1e-09
    # lower bound: 1
    population['hill'][np.flatnonzero(population['hill'] < 1)] = 1
    if np.any(population['hill'] <= 0):
        raise ValueError("population['hill'] is lower than or equal to zero.")


def reproduction(population):
    """produces next generation via mutation and natural selection"""
    
    mutation(population)
    population['phenotype'] = phenotype(population)
    if np.any(population['phenotype'] < 0):
        raise ValueError("population['phenotype'] is lower than zero.")
    
    population['fitness'] = fitness(population)
    if np.any(population['fitness'] < 0):
        raise ValueError("population['fitness'] is negative.")
    
    mean_fitness_before = np.mean(population['fitness'])
    #print(f"Mean fitness before selection round {t}:", mean_fitness_before)
    if (mean_fitness_before == 0):
        raise RuntimeError("Mean fitness of population decreased to 0.")
    else:
        offspring = np.random.poisson(population['fitness'] / mean_fitness_before)
    #print("Pop before reproduction: \n", population)
    population = np.repeat(population, offspring)
    if len(population) > n:
        population = np.random.choice(population, n, replace=False)
    elif len(population) < n:
        clones = np.random.choice(np.arange(len(population)), n-len(population), 
                                  replace=False)
        clone_rep = np.ones(len(population), dtype=np.int)
        clone_rep[clones] = 2
        population = np.repeat(population, clone_rep)
    return population


# ITERATION

if __name__ == '__main__':

    population = create_population(n)
    # test with constant environment
    #enviro = environment(0)
    #population['E'] = enviro[0]
    #population['C'] = enviro[1]
    
    list_environment = []
    list_C = []
    list_mean_fitness = []
    list_mean_b = []
    list_mean_phenotype = []
    
    for t in progressbar.progressbar(range(no_gens+1)):
        enviro = environment(t)
        population['E'] = enviro[0]
        population['C'] = enviro[1]
        population = reproduction(population)
        
        ##### Environment
        mean_environment = np.mean(population['E'])
        list_environment.append(mean_environment)
        
        mean_C = np.mean(population['C'])
        list_C.append(mean_C)
        
        #print(f"Population after selection round {t}: \n", population)
        #print()
        mean_fitness_after = np.mean(population['fitness'])
        #print(f"Mean fitness after selection round {t}:", mean_fitness_after)
        list_mean_fitness.append(mean_fitness_after)
        
        ##### Population mean of b
        mean_b = np.mean(population['b'])
        list_mean_b.append(mean_b)
        
        #### Population mean of phenotype
        mean_phenotype = np.mean(population['phenotype'])
        list_mean_phenotype.append(mean_phenotype)
    
    
    # OUTPUT
    print()
    
    # check that time counter t has reached maximum number of generations
    if t != no_gens:
        raise ValueError("t does not equal the total number of generations.")
    
    # checking C
    values_C, counts_C = np.unique(population['C'], return_counts=True)
    if sum(counts_C) != n:
        raise ValueError("The number of C values does not equal n.")
    table_C = np.zeros(len(values_C), dtype=[('C values','<f8'), ('C counts','<i8')])
    table_C['C values'] = values_C
    table_C['C counts'] = counts_C
    print("C table:", table_C)
    
    # checking G
    values_G, counts_G = np.unique(population['G'], return_counts=True)
    if sum(counts_G) != n:
        raise ValueError("The number of G values does not equal n.")
    table_G = np.zeros(len(values_G), dtype=[('G values','<f8'), ('G counts','<i8')])
    table_G['G values'] = values_G
    table_G['G counts'] = counts_G
    print("G table:", table_G)
    
    # checking b
    values_b, counts_b = np.unique(population['b'], return_counts=True)
    if sum(counts_b) != n:
        raise ValueError("The number of b values does not equal n.")
    table_b = np.zeros(len(values_b), dtype=[('b values','<f8'), ('b counts','<i8')])
    table_b['b values'] = values_b
    table_b['b counts'] = counts_b
    print("b table:", table_b)
    
    # checking alpha
    values_alpha, counts_alpha = np.unique(population['alpha'], return_counts=True)
    if sum(counts_alpha) != n:
        raise ValueError("The number of alpha values does not equal n.")
    table_alpha = np.zeros(len(values_alpha), dtype=[('alpha values','<f8'), ('alpha counts','<i8')])
    table_alpha['alpha values'] = values_alpha
    table_alpha['alpha counts'] = counts_alpha
    print("alpha table:", table_alpha)
    
    # checking hill
    values_hill, counts_hill = np.unique(population['hill'], return_counts=True)
    if sum(counts_hill) != n:
        raise ValueError("The number of hill values does not equal n.")
    table_hill = np.zeros(len(values_hill), dtype=[('hill values','<f8'), ('hill counts','<i8')])
    table_hill['hill values'] = values_hill
    table_hill['hill counts'] = counts_hill
    print("hill table:", table_hill)
    
    # checking phenotype
    values_phenotype, counts_phenotype = np.unique(population['phenotype'], return_counts=True)
    if sum(counts_phenotype) != n:
        raise ValueError("The number of phenotype values does not equal n.")
    table_p = np.zeros(len(values_phenotype), dtype=[('p values','<f8'), ('p counts','<i8')])
    table_p['p values'] = values_phenotype
    table_p['p counts'] = counts_phenotype
    print("p table:", table_p)
    
    # checking fitness
    values_fitness, counts_fitness = np.unique(population['fitness'], return_counts=True)
    if sum(counts_fitness) != n:
        raise ValueError("The number of fitness values does not equal n.")
    table_fitness = np.zeros(len(values_fitness), dtype=[('fitness values', '<f8'), ('fitness counts', '<i8')])
    table_fitness['fitness values'] = values_fitness
    table_fitness['fitness counts'] = counts_fitness
    print("fitness table:", table_fitness)
    
    #print(population)
    
    print()
    """
    # advanced plotting
    space = np.linspace(0, 1, 100)
    phenotype_container = []
    for C in space:
        population['basal'] = population['G'] + population['b'] * C#population['C']
        def fun(x):
            x = np.maximum(1e-8, x)
            return (population['basal'] - x + population['alpha']
                    * (x ** population['hill'] / (1 + x ** population['hill'])))
        sol = optimize.root(fun, np.zeros(n), method='excitingmixing', options=dict(fatol=1e-8))
        phenotype_plot = sol.x
        phenotype_container.append(phenotype_plot)
    plt.plot(space, phenotype_container, color='k', linewidth=0.1)
    plt.ylim(-0.05, 2.55)
    """
    
    #profile.print_stats()
    
    
    """
    # plotting
    plt.plot(list_environment, label='environment')
    plt.ylim(-0.05, 1.05)
    
    plt.plot(list_C, label='C')
    plt.ylim(-0.05, 1.05)
    
    plt.plot(list_mean_fitness, label='fitness')
    plt.ylim(-0.05, 1.05)
    
    plt.plot(list_mean_b, label='b')
    plt.ylim(-0.05, 1.05)
    
    plt.plot(list_mean_phenotype, label='phenotype')
    plt.ylim(-0.05, 1.05)
    
    plt.legend()
    """
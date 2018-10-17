#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:01:45 2018

@author: romanzug
"""

import numpy as np

def create_population(n):
    pop = np.zeros(n, dtype=[('a','<f8'),('b','<f8'),('position','<i8'),('supergene','<f8')])
    for f in pop.dtype.fields:
        pop[f] = np.random.randn(n)*5
    return pop

#pop = create_population(10)
#print(pop["position"])
#print(pop["supergene"])

def mutate(population):
    n = len(population)
    population["a"] += np.random.randn(n)
    population["b"] += 2.5*np.random.randn(n) + 0.25
    population["supergene"] += np.random.randn(n)
    population["position"] += np.random.randint(-1,2,size=n)
    
def fitness(population):
    fit_fun = lambda x,t: np.exp(-np.power(population[x]-t,2))
    return fit_fun("a",15) + fit_fun("b",2.5) \
             + fit_fun("supergene",10) + fit_fun("position",0)

def do_step(population):
    n = len(population)
    for _ in range(5):
        mutate(population)
    fitnesses = fitness(population)
    offspring = np.random.poisson(fitnesses / np.mean(fitnesses))
    population = np.repeat(population,offspring)
    if len(population) > n:
        population = np.random.choice(population,n,replace=False)
    elif len(population) < n:
        clones = np.random.choice(np.arange(len(population)),n-len(population),replace=False)
        clone_rep = np.ones(len(population), dtype=np.int)
        clone_rep[clones] = 2
        population = np.repeat(population,clone_rep)
    return population

pop = create_population(8)
for _ in range(10):
    pop = do_step(pop)
print(pop)
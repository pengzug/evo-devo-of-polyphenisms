#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:21:26 2018

@author: romanzug
"""

############################
# evo-devo of polyphenisms #
############################

import numpy as np
#import matplotlib.pyplot as plt
from scipy.integrate import odeint

####
# 0. general settings

#t = 1 # current time (number of generations)
t_end = 3 # endpoint of evolutionary process (maximum number of generations)

I0 = np.random.uniform(0, 1) # baseline phenotype

#print("Generation: " + str(t))

def main_function(t_end): # for a single individual
    for t in range(1, t_end + 1):
        #mutation_rate = 0.001
        #mutational_step = np.random.normal(0, 0.05)
        #I0 += I
        
        print("Generation: " + str(t))
        print("Environmental value: " + str(environ_variation(t)))
        print("Environmental cue:   " + str(environ_cue(t)))
        print("Phenotype:           " + str(phenotype(t)))
        print() # prints an empty line
    

####
# 1. environmental variation
# Testläufe: testing_env_var.py (ehemals botero.py)

# returns value of environmental variation at generation t
def environ_variation(t):
    A = 1 # scaling constant for deterministic factors
    B = 1 - A # scaling constant for stochastic factors
    L = 1 # number of time steps per generation (= life span)
    R = 25 # relative timescale of environmental variation (no. of gen.s per environmental cycle)
    # eps: stochastic error term
    
    accuracy = 10 * t + 1
    
    ti = np.linspace(0, t, accuracy)
    E_vorl = []
    for step in range(len(ti)):
        eps = np.random.uniform(0, 1)
        var = A * (np.sin(2 * np.pi * ti[step] / (L * R)) + 1) / 2 + B * eps
        E_vorl.append(var)

    # the list E only contains the values for each generation
    fra = int((accuracy - 1) / t)
    E = E_vorl[0::fra]
    Et = E[t]
    
    return Et

#print("Environmental value: " + str(environ_variation(t)))


####
# 2. environmental cue

# returns value of environmental cue at generation t
def environ_cue(t):
    hurz = environ_variation(t)
    P = 0.9 # environmental predictability
    #mu = P * hurz # for [-1, 1] range of E and C
    mu = 0.5 * (1 - P) + P * hurz # for [0, 1] range of E and C
    #sigma = (1 - P) / 3 # for [-1, 1] range of E and C
    sigma = 0.5 * (1 - P) / 3 # for [0, 1] range of E and C
       
    cue = np.random.normal(mu, sigma) # cues are drawn from a normal distr with mean mu, SD sigma
    
    return cue

#print("Environmental cue:   " + str(environ_cue(t)))


####
# 3. phenotype
# Testläufe: testing_dgl_1, testing_dgl_2

# returns phenotypic value at generation t
def phenotype(t):
    def dgl(x, tu):
        env_cue = environ_cue(t)
        #I0 = np.random.uniform(0, 1) # baseline phenotype
        alpha = 0 # strength of the feedback
        n = 1 # Hill coefficient
        dxdt = env_cue - x + alpha * (x ** n / (1 + x ** n))
        
        return dxdt
    
    x0 = 0
    t_hilla = 1000 # hier hoffen wir, dass t_hilla ausreicht, um equi zu finden
    tu = np.linspace(0, t_hilla, t_hilla)
    x = odeint(dgl, x0, tu)
    
    # finding the steady state of the ode
    hui = x[t_hilla - 1] # hui is a single-item list
    [hus] = hui # how to get the item from a single-item list
    
    return hus

#print("Phenotype:           " + str(phenotype(t)))



####
# Finally calling the main function

main_function(t_end)
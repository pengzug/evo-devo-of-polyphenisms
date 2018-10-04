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

# Importing

import numpy as np
#from scipy.integrate import ode#int
from scipy.integrate import solve_ivp
#import matplotlib.pyplot as plt



# Parameters

A = 1       # scaling constant for deterministic factors
B = 1 - A   # scaling constant for stochastic factors
L = 1       # number of time steps per generation (= life span)
R = 25      # relative timescale of environmental variation (no. of gen.s per environmental cycle)
P = 0.5       # environmental predictability
#alpha = 0   # strength of the feedback
#hill = 1    # Hill coefficient


def environ_variation(t):
    """returns environmental value E at generation t"""
    eps = np.random.uniform(0, 1)   # stochastic error term
    E = A * (np.sin(2 * np.pi * t / (L * R)) + 1) / 2 + B * eps
    return E
    
def environ_cue(t):
    """returns environmental cue C at generation t"""
    E = environ_variation(t)
    mu = 0.5 * (1 - P) + P * E
    sigma = 0.5 * (1 - P) / 3
    C = np.random.normal(mu, sigma)
    return C

#def population(n, t):
#    Cu = environ_cue(t)
#    return Cu


def population(n, t):
    """creates a population of n individuals at generation t,
       where each individual receives an environmental cue C which is correlated  
       with environmental condition E"""
    
    # E: environmental condition
    eps = np.random.uniform(0, 1)   # stochastic error term
    E = A * (np.sin(2 * np.pi * t / (L * R)) + 1) / 2 + B * eps
    print("Environmental value E:     " + str(E))
    
    # C: environmental cue
    mu = 0.5 * (1 - P) + P * E
    sigma = 0.5 * (1 - P) / 3
    C = np.random.normal(mu, sigma)
    print("Environmental cue C:       " + str(C))
    
    # pop: population
    pop = np.zeros(n, dtype=[('alpha','<f8'), ('hill','<f8'), ('phenotype','<f8')])
    pop['hill'] = 1
    def phenotype_ode(ta, x):
        """defines the differential equation(s)"""
        dxdt = C - x + pop['alpha'] * (x ** pop['hill'] / (1 + x ** pop['hill']))
        return dxdt
    #t_end = 100
    
    # diese Version funktioniert! (bis auf dass es ab ca. 10. NKS nicht übereinstimmt
    odin = solve_ivp(phenotype_ode, [0, 100], [0, 1], method='BDF')
    print(odin.y[0])
    print(odin.y[1])
    #print("last: " + str(odin.y[0][-1]))
    #print("2.last: " + str(odin.y[0][-2]))
    last_diff = abs(odin.y[1][-1] - odin.y[1][-2])
    #print("last_diff: " + str(last_diff))
    if (last_diff < 1e-7) != 1:
    #if sum(last_diff > 1e-7) != 0: # deaktiviert wg "TypeError: 'numpy.bool_' object is not iterable"
        print("Could not find the equilibrium! Probably t_end too low...")
    steady_state = odin.y[0][-1] #/ t_end # / t_end nötig wg BDF
    pop['phenotype'] = steady_state
    return pop['phenotype']

#population = population(1)


for t in range(1):
    print("Generation: " + str(t))
    #print("Environmental value E:     " + str(E))
    #print("Environmental cue C:       " + str(C))
    print("soll identisch sein mit C: " + str(population(1, t)))
    print()

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
P = 0.2     # environmental predictability
#alpha = 0  # strength of the feedback
#hill = 1   # Hill coefficient

def population(n, t):
    """creates a population of n individuals at generation t,
       where each individual receives an environmental cue C 
       which is correlated with environmental condition E"""
    
    # E: environmental condition
    eps = np.random.uniform(0, 1)   # stochastic error term
    E = A * (np.sin(2 * np.pi * t / (L * R)) + 1) / 2 + B * eps
    print("Env value E: " + str(E))
    
    # C: environmental cue
    mu = 0.5 * (1 - P) + P * E
    sigma = 0.5 * (1 - P) / 3
    C = np.random.normal(mu, sigma)
    print("Env cue C:   " + str(C))
    
    # pop: population
    pop = np.zeros(n, dtype=[('alpha','<f8'), ('hill','<f8'), ('phenotype','<f8'), ('fitness','<f8')])
    pop['hill'] = 1
    #pop['alpha'] = 1
    
    def phenotype_ode(ta, x):
        """defines the differential equation(s)"""
        dxdt = C - x + pop['alpha'] * (x ** pop['hill'] / (1 + x ** pop['hill']))
        return dxdt
    t_end = 1e06
    odin(nu) = solve_ivp(phenotype_ode, [0, t_end], [0, 1], method='BDF')
    
    """
    print(odin.y[0])
    print(odin.y[1])
    last_diff_lower = abs(odin.y[0][-1] - odin.y[0][-2])
    last_diff_upper = abs(odin.y[1][-1] - odin.y[1][-2])
    #print("last_diff_lower: " + str(last_diff_lower))
    #print("last_diff_upper: " + str(last_diff_upper))
    if (last_diff_lower < 1e-7) != 1:
        print("Could not find the lower equilibrium! Probably t_end too low...")
    if (last_diff_upper < 1e-7) != 1:
        print("Could not find the upper equilibrium! Probably t_end too low...")
    steady_state_lower = odin.y[0][-1]
    steady_state_upper = odin.y[1][-1]
    print("lower equi:  " + str(steady_state_lower))
    print("upper equi:  " + str(steady_state_upper))
    diff_upper_lower = abs(steady_state_lower - steady_state_upper)
    if (diff_upper_lower < 1e-05) != 1:
        print("steady states do not match!")
    pop['phenotype'] = steady_state_lower
    
    #def fitness_function():
    
    return pop['phenotype']
    """

#print(population(1, 24))

t = 4
#for t in range(3):
print("Generation: " + str(t))
print("Phenotype:  " + str(population(1, t)))
print()
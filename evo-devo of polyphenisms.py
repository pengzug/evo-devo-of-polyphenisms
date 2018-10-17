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
P = 1     # environmental predictability

def population(n, t):
    """creates a population of n individuals at generation t,
       where each individual receives an environmental cue C 
       which is correlated with environmental condition E"""
    
    # E: environmental condition (e.g. temperature)
    eps = np.random.uniform(0, 1)   # stochastic error term
    E = A * (np.sin(2 * np.pi * t / (L * R)) + 1) / 2 + B * eps
    print("Env value E: " + str(E))
    
    # C: environmental cue (e.g. photoperiod)
    mu = 0.5 * (1 - P) + P * E
    sigma = 0.5 * (1 - P) / 3
    C = np.random.normal(mu, sigma)
    print("Env cue C:   " + str(C))
    
    # pop: population
    # alpha: strength of the feedback (in the ODE); initial value: 0
    # hill: Hill coefficient (in the ODE); initial value: 1
    pop = np.zeros(n, dtype=[('alpha','<f8'), ('hill','<f8'), ('phenotype','<f8')])
    pop['alpha'] = np.random.rand(n) * 2        # 1.4
    print("alpha: ", pop['alpha'])
    pop['hill']  = np.random.randint(1, 11, n)  # 8
    print("hill: ", pop['hill'])
    def phenotype(n):
        """creates the phenotype"""
        def phenotype_ode(t_ode, x):
            """defines the ode(s) for the phenotype"""
            dxdt = C - x + pop['alpha'] * (x ** pop['hill'] / (1 + x ** pop['hill']))
            return dxdt
        t_end = 1e09
        sol_lower = solve_ivp(phenotype_ode, [0, t_end], np.zeros(n), method='BDF')
        sol_upper = solve_ivp(phenotype_ode, [0, t_end], np.ones(n) * 10, method='BDF')
        lower_steady_states = sol_lower.y[:, -1]
        upper_steady_states = sol_upper.y[:, -1]
        print(lower_steady_states)
        print(upper_steady_states)
        ### WARNING: ALWAYS CHOOSING UPPER STEADY STATE ###
        return upper_steady_states
    pop['phenotype'] = phenotype(n)
    return pop
    
    
    """
    print(sol.y[0])
    print(sol.y[1])
    last_diff_lower = abs(sol.y[0][-1] - sol.y[0][-2])
    last_diff_upper = abs(sol.y[1][-1] - sol.y[1][-2])
    #print("last_diff_lower: " + str(last_diff_lower))
    #print("last_diff_upper: " + str(last_diff_upper))
    if (last_diff_lower < 1e-7) != 1:
        print("Could not find the lower equilibrium! Probably t_end too low...")
    if (last_diff_upper < 1e-7) != 1:
        print("Could not find the upper equilibrium! Probably t_end too low...")
    lower_equilibrium = sol.y[0][-1]
    upper_equilibrium = sol.y[1][-1]
    print("lower equi:  " + str(lower_equilibrium))
    print("upper equi:  " + str(upper_equilibrium))
    diff_upper_lower = abs(lower_equilibrium - upper_equilibrium)
    if (diff_upper_lower < 1e-05) != 1:
        print("steady states do not match!")
    pop['phenotype'] = lower_equilibrium
    """

n = 4
t = 2

print("Population size: ", n)
print("Generation: ", t)
print("Population:  ", population(4, t))

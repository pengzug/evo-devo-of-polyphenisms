#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:58:42 2018

@author: romanzug
"""

import numpy as np
from scipy.integrate import odeint
#from scipy.optimize import fsolve
#import matplotlib.pyplot as plt

def population(n):
    """creates a population of n individuals at generation t"""
    pop = np.zeros(n, dtype=[('alpha','<f8'), ('hill','<f8'), ('phenotype','<f8')])
    pop['hill'] = 1
    def phenotype_ode(x, t):
        dxdt = 0.6243449435824274 - x + pop['alpha'] * (x ** pop['hill'] / (1 + x ** pop['hill']))
        return dxdt
    init_cond = np.zeros(n)
    time_frame = np.linspace(0, 100, 100)
    sol = odeint(phenotype_ode, init_cond, time_frame)
    
    last_diff = abs(sol[-1, :] - sol[-2, :])
    if sum(last_diff > 1e-8) != 0:
        print("Could not find the equilibrium! Probably time too short...")
    steady_state = sol[-1, :]
    pop['phenotype'] = steady_state
    #return pop
    return pop['phenotype']

population = population(1)
print(population)

#equi = fsolve(phenotype_ode, putative_equi, args=None)
#print(equi)


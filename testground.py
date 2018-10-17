#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:29:33 2018

@author: romanzug
"""

import numpy as np
from scipy.integrate import solve_ivp

def create_population(n):
    pop = np.zeros(n, dtype=[('alpha','<f8'),('beta','<f8'),('phenotype','<f8')])
    pop['alpha'] = np.random.rand(n) * 2
    print("alpha: ", pop['alpha'])
    pop['beta'] = np.random.randint(1, 11, n)
    print("beta: ", pop['beta'])
    def phenotype(n):
        #phenotype_simple = 2 * pop['alpha'] + pop['beta']
        #print("alpha length: " + str(len(pop['alpha'])))
        #print("beta length :" + str(len(pop['beta'])))
        #return phenotype_simple
        def phenotype_ode(t_ode, y):
            dydt = 0.5 - y + pop['alpha'] * (y ** pop['beta'] / (1 + y ** pop['beta']))
            #print(len(y))
            #print("alpha length: " + str(len(pop['alpha'])))
            #print("beta length :" + str(len(pop['beta'])))
            return dydt
        t_end = 1e09
        sol_lower = solve_ivp(phenotype_ode, [0, t_end], np.zeros(n), method='BDF')
        sol_upper = solve_ivp(phenotype_ode, [0, t_end], np.ones(n) * 10, method='BDF')
        lower_steady_states = sol_lower.y[:, -1]
        upper_steady_states = sol_upper.y[:, -1] # last entry is assumed to be the steady state
        print("lower st st: ", lower_steady_states)
        print("upper st st: ", upper_steady_states)
        ### WARNING: ALWAYS CHOOSING UPPER STEADY STATE ###
        return upper_steady_states 
    pop['phenotype'] = phenotype(n)
    return pop

popul = create_population(4)
print(popul)
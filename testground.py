#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:29:33 2018

@author: romanzug
"""

import numpy as np
#from scipy.integrate import solve_ivp

def create_population(n):
    pop = np.zeros(n, dtype=[('alpha','<f8'),('hill','<f8'),('phenotype','<f8')])
    pop['alpha'] = np.random.randn(n)
    pop['hill'] = 7
    pop['phenotype'] = 2 * pop['alpha'] + pop['hill']
    #for f in pop.dtype.fields:
    #    pop[f] = np.random.randn(n)#*5
    return pop

pop = create_population(10)
print(pop)
odie = pop['phenotype']
print(odie)
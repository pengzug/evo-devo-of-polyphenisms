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
#import matplotlib.pyplot as plt
#from scipy.integrate import odeint


# Parameters

A = 1       # scaling constant for deterministic factors
B = 1 - A   # scaling constant for stochastic factors
L = 1       # number of time steps per generation (= life span)
R = 25      # relative timescale of environmental variation (no. of gen.s per environmental cycle)
P = 1       # environmental predictability
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

def population(n, t):
    pop = np.zeros(n, dtype=[('alpha','<f8'), ('hill','<f8')])
    pop['hill'] = 1     # initial Hill coefficient
    return pop

pop = population(5, 1)
print(pop)

#def phenotype(t):

"""
for t in range(1, 4):
    print("Generation: " + str(t))
    print("Environmental value: " + str(environ_variation(t)))
    print("Environmental cue:   " + str(environ_cue(t)))
    print()
"""
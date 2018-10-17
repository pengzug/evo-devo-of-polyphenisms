#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:59:02 2018

@author: romanzug
"""

from scipy.integrate import solve_ivp
#import numpy as np
import matplotlib.pyplot as plt

#n = 3
def test_ode(t, y):
    dydt = 0.4 - y + 1.2 * (y ** 4 / (1 + y ** 4))
    return dydt
sol = solve_ivp(test_ode, [0, 1e07], [0, 1, 10, 20], method='BDF')
print(sol.y)
#plt.plot(sol.t, sol.y[3])
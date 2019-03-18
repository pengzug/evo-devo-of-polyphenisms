#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#####################################################
#
#       edop.py
#
#       Author: Roman Zug (roman.zug@biol.lu.se)
#
#####################################################
"""

import pprint

import numpy as np
from scipy import optimize
import tqdm
import click
from loguru import logger


def environment(t):
    """sets environmental variation and calculates corresponding cue"""

    # E: environmental condition (e.g. temperature)
    eps = 1 - np.random.random() # stochastic error term
    E = A * (np.sin(2 * np.pi * t / (L * R)) + 1) / 2 + B * eps

    # C: environmental cue (e.g. photoperiod)
    mu_env = 0.5 * (1 - P) + P * E
    sigma_env = 0.5 * (1 - P) / 3
    C = np.random.normal(mu_env, sigma_env)
    if C <= 0.0:
        C = 1e-09
    if C <= 0.0:
        raise ValueError("C is lower than or equal to zero.")

    return E, C


def create_population(n):
    """creates the starting population of n individuals"""
    # E: environmental condition
    # C: environmental cue
    # G: genetic component of basal expression of x
    # b: degree to which basal expression of x depends on environment
    # basal: basal expression of the lineage factor x (basal = G + b*C)
    # alpha: strength of the feedback (in the ODE); initial value: nicht mehr 1e-09, sondern 1
    # hill: Hill coefficient (in the ODE); initial value: 1
    population = np.zeros(n, dtype=[('E','<f8'), ('C','<f8'), ('G','<f8'), ('b', '<f8'), 
                                    ('basal','<f8'), ('alpha','<f8'), ('hill','<f8'), 
                                    ('phenotype','<f8'), ('fitness','<f8')])
    population['G'] = 1 - np.random.random(n)
    population['b'] = 1 - np.random.random(n)
    population['alpha'] = 1
    population['hill'] = 1
    return population


# phenotype function with scipy.optimize.root
def phenotype(population):
    """creates the phenotype"""

    population['basal'] = population['G'] + population['b'] * population['C']

    if np.any(population['basal'] <= 0):
        raise ValueError("population['basal'] is lower than or equal to zero.")

    if np.any(population['alpha'] <= 0):
        raise ValueError("population['alpha'] is lower than or equal to zero before mutation.")

    def fun(x):
        x = np.maximum(1e-8, x)
        return (population['basal'] - x + population['alpha'] 
                * (x ** population['hill'] / (1 + x ** population['hill'])))
    
    sol = optimize.root(fun, np.zeros(n), method='excitingmixing', options=dict(fatol=1e-8))
    return sol.x


def fitness(population):
    """calculates the fitness"""
    
    mean_phenotype = np.mean(population['phenotype'])
    # fitness_part_1: fitness component deriving from phenotypic (mis)match
    fitness_part_1 = (np.exp(-np.abs(population['E'] - population['phenotype']) * tau))
    # fitness_part_2: fitness component deriving from disruptive selection (advantage of rare phenotypes)
    fitness_part_2 = (1 - np.exp(-np.abs(mean_phenotype - population['phenotype']) * phi))
    W = (fitness_part_1 * fitness_part_2) - kd
    W[np.flatnonzero(W < 0.0)] = 0.0
    return W


def mutation(population):
    """introduces random mutations for the genetic parameters"""
    
    n = len(population)
    
    # mutation of G; 1e-09 <= G <= 1
    indices_muta_G, = np.where(np.random.random(n) <= mu) # comma is important!
    number_muta_G = indices_muta_G.size
    population['G'][indices_muta_G] += np.random.normal(0, 0.05, number_muta_G)
    # lower bound: 1e-09
    non_pos_indices_G, = np.where(population['G'] <= 0.0) # comma is important!
    population['G'][non_pos_indices_G] = 1e-09
    # upper bound: 1
    population['G'][np.flatnonzero(population['G'] > 1)] = 1
    if np.any(population['G'] <= 0):
        raise ValueError("population['G'] is lower than or equal to zero.")
        
    # mutation of b; 1e-09 <= b <= 1
    indices_muta_b, = np.where(np.random.random(n) <= mu)
    number_muta_b = indices_muta_b.size
    population['b'][indices_muta_b] += np.random.normal(0, 0.05, number_muta_b)
    # lower bound: 1e-09
    non_pos_indices_b, = np.where(population['b'] <= 0.0)
    population['b'][non_pos_indices_b] = 1e-09
    # upper bound: 1
    population['b'][np.flatnonzero(population['b'] > 1)] = 1
    if np.any(population['b'] <= 0):
        raise ValueError("population['b'] is lower than or equal to zero.")
    
    # mutation of alpha; lower bound either 1e-09 or 1
    indices_muta_alpha, = np.where(np.random.random(n) <= mu)
    number_muta_alpha = indices_muta_alpha.size
    population['alpha'][indices_muta_alpha] += np.random.normal(0, 0.05, number_muta_alpha)
    # lower bound: 1e-09
    #non_pos_indices_alpha, = np.where(population['alpha'] <= 0.0)
    #population['alpha'][non_pos_indices_alpha] = 1e-09
    # lower bound: 1
    population['alpha'][np.flatnonzero(population['alpha'] < 1)] = 1
    if np.any(population['alpha'] <= 0):
        raise ValueError("population['alpha'] is lower than or equal to zero after mutation.")
    
    # mutation of hill; lower bound either 1e-09 or 1
    indices_muta_hill, = np.where(np.random.random(n) <= mu)
    number_muta_hill = indices_muta_hill.size
    population['hill'][indices_muta_hill] += np.random.normal(0, 0.05, number_muta_hill)
    # lower bound: 1e-09
    #non_pos_indices_hill, = np.where(population['hill'] <= 0.0)
    #population['hill'][non_pos_indices_hill] = 1e-09
    # lower bound: 1
    population['hill'][np.flatnonzero(population['hill'] < 1)] = 1
    if np.any(population['hill'] <= 0):
        raise ValueError("population['hill'] is lower than or equal to zero.")


def reproduction(population):
    """produces next generation via mutation and natural selection"""
    
    mutation(population)
    population['phenotype'] = phenotype(population)
    if np.any(population['phenotype'] < 0):
        raise ValueError("population['phenotype'] is lower than zero.")
    
    population['fitness'] = fitness(population)
    if np.any(population['fitness'] < 0):
        raise ValueError("population['fitness'] is negative.")
    
    mean_fitness_before = np.mean(population['fitness'])
    #print(f"Mean fitness before selection round {t}:", mean_fitness_before)
    if (mean_fitness_before == 0):
        raise RuntimeError("Mean fitness of population decreased to 0.")
    else:
        offspring = np.random.poisson(population['fitness'] / mean_fitness_before)
    #print("Pop before reproduction: \n", population)
    population = np.repeat(population, offspring)
    if len(population) > n:
        population = np.random.choice(population, n, replace=False)
    elif len(population) < n:
        clones = np.random.choice(np.arange(len(population)), n-len(population), 
                                  replace=False)
        clone_rep = np.ones(len(population), dtype=np.int)
        clone_rep[clones] = 2
        population = np.repeat(population, clone_rep)
    return population


def main(n=500, num_gens=1000, a=1, b=0, l=1, r=500,
         p=0.9, tau=1, phi=4, kd=0.01, mu=0.001):
    logger.add(
        sys.stdout, colorize=True,
        format="<green>{time}</green> <level>{message}</level>"
    )

    logger.info("Population size: {}", n)
    logger.info("Generations: {}", num_gens)
    logger.info("A: {}", a)
    logger.info("B: {}", b)
    logger.info("L: {}", l)
    logger.info("log R: {}", np.log10(r))
    logger.info("P: {}", p)
    logger.info("tau: {}", tau)
    logger.info("phi: {}", phi)
    logger.info("kd: {}", kd)
    logger.info("mu: {}", mu)

    population = create_population(n)

    list_environment = []
    list_C = []
    list_mean_fitness = []
    list_mean_b = []
    list_mean_phenotype = []

    for t in tqdm.tqdm(range(1, no_gens + 1)):
        enviro = environment(t)
        population['E'] = enviro[0]
        population['C'] = enviro[1]
        population = reproduction(population)

        # Environment
        mean_environment = np.mean(population['E'])
        list_environment.append(mean_environment)

        mean_C = np.mean(population['C'])
        list_C.append(mean_C)

        mean_fitness_after = np.mean(population['fitness'])
        list_mean_fitness.append(mean_fitness_after)

        # Population mean of b
        mean_b = np.mean(population['b'])
        list_mean_b.append(mean_b)

        # Population mean of phenotype
        mean_phenotype = np.mean(population['phenotype'])
        list_mean_phenotype.append(mean_phenotype)

    # OUTPUT
    for var in ('C', 'G', 'b', 'alpha', 'hill', 'phenotype', 'fitness'):
        values, counts = np.unique(population[var], return_counts=True)
        if np.sum(counts) != n:
            raise ValueError("The number of C values does not equal n.")
        table = np.zeros(
            len(values),
            dtype=[(f'{var} values', '<f8'), (f'{var} counts', '<i8')]
        )
        table[f'{var} values'] = values
        table[f'{var} counts'] = counts
        formatted_table = pprint.pformat(table)
        logger.info(f"{var} table: {formatted_table}")

    return population

    """
    # advanced plotting
    space = np.linspace(0, 1, 100)
    phenotype_container = []
    for C in space:
        population['basal'] = population['G'] + population['b'] * C#population['C']
        def fun(x):
            x = np.maximum(1e-8, x)
            return (population['basal'] - x + population['alpha']
                    * (x ** population['hill'] / (1 + x ** population['hill'])))
        sol = optimize.root(fun, np.zeros(n), method='excitingmixing', options=dict(fatol=1e-8))
        phenotype_plot = sol.x
        phenotype_container.append(phenotype_plot)
    plt.plot(space, phenotype_container, color='k', linewidth=0.1)
    plt.ylim(-0.05, 2.55)
    """

    """
    # plotting
    plt.plot(list_environment, label='environment')
    plt.ylim(-0.05, 1.05)

    plt.plot(list_C, label='C')
    plt.ylim(-0.05, 1.05)

    plt.plot(list_mean_fitness, label='fitness')
    plt.ylim(-0.05, 1.05)

    plt.plot(list_mean_b, label='b')
    plt.ylim(-0.05, 1.05)

    plt.plot(list_mean_phenotype, label='phenotype')
    plt.ylim(-0.05, 1.05)

    plt.legend()
    """


@click.command()
@click.option('-n', help='Population size', default=500, type=int)
@click.option('-N', '--num-gens', help='Numer of generations', default=1000, type=int)
@click.option('-a', help='Scaling constant for deterministic environmental actors',
              default=1, type=float)
@click.option('-b', help='Scaling constant for stochastic environmental factors',
              default=0, type=float)
@click.option('-l', help='Number of time steps per generation (life span)',
              default=1, type=int)
@click.option('-r', help=(
    'Relative time scale of environmental variation '
    '(number of generations per environmental cycle)'
), default=500, type=float)
@click.option('-p', help='Environmental predictability', default=0.9, type=float)
@click.option('--tau', help='Strength of fitness decay due to phenotypic mismatch',
              default=1, type=float)
@click.option('--phi', help='Strength of fitness gain due to advantage of rare/extreme phenotypes',
              default=4, type=float)
@click.option('--kd', help='Cost for developmental plasticity',
              default=0.01, type=float)
@click.option('--mu', help='Mutation rate', default=0.001, type=float)
def cli(**kwargs):
    return main(**kwargs)


if __name__ == '__main__':
    cli()
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

import sys
import pprint
import contextlib
import multiprocessing
import concurrent.futures

import toml
import numpy as np
from scipy import optimize
import tqdm
import click
from loguru import logger

logger.remove()

config = dict(
    n=500,  # Population size
    num_gen=1000,  # Number of generations
    A=1,  # Scaling constant for deterministic environmental actors
    B=0,  # Scaling constant for stochastic environmental factors
    L=1,  # Number of time steps per generation (life span)

    # Relative time scale of environmental variation
    # (number of generations per environmental cycle)
    R=500,

    P=0.9,  # Environmental predictability
    tau=1,  # Strength of fitness decay due to phenotypic mismatch
    phi=4,  # Strength of fitness gain due to advantage of rare/extreme phenotypes
    kd=0.01,  # Cost for developmental plasticity
    mu=0.001  # Mutation rate
)


def environment(t):
    """sets environmental variation and calculates corresponding cue"""
    # E: environmental condition (e.g. temperature)
    eps = 1 - np.random.random()  # stochastic error term
    E = (
        config['A'] * (np.sin(2 * np.pi * t / (config['L'] * config['R'])) + 1)
        / 2 + config['B'] * eps
    )

    # C: environmental cue (e.g. photoperiod)
    mu_env = 0.5 * (1 - config['P']) + config['P'] * E
    sigma_env = 0.5 * (1 - config['P']) / 3
    C = max(1e-9, np.random.normal(mu_env, sigma_env))
    assert C > 0
    return E, C


def create_population(n):
    """creates the starting population of n individuals"""
    # E: environmental condition
    # C: environmental cue
    # G: genetic component of basal expression of x
    # b: degree to which basal expression of x depends on environment
    # basal: basal expression of the lineage factor x (basal = G + b*C)
    # alpha: strength of the feedback (in the ODE); initial value: 1
    # hill: Hill coefficient (in the ODE); initial value: 1
    population = np.zeros(n, dtype=[
        ('E', '<f8'), ('C', '<f8'), ('G', '<f8'), ('b', '<f8'),
        ('basal', '<f8'), ('alpha', '<f8'), ('hill', '<f8'),
        ('phenotype', '<f8'), ('fitness', '<f8')
    ])
    population['G'] = 1 - np.random.random(n)
    population['b'] = 1 - np.random.random(n)
    population['alpha'] = 1
    population['hill'] = 1
    return population


def find_roots(fun, limits, args=None):
    if args is None:
        args = tuple()

    search_space = np.logspace(np.log10(limits[0]), np.log10(limits[1]), 100)
    val = fun(search_space, *args)
    sign_changes = np.sign(val[1:]) != np.sign(val[:-1])

    brackets = zip(
        search_space[:-1][sign_changes],
        search_space[1:][sign_changes]
    )
    roots = []

    for bracket in brackets:
        # polish initial brackets to convergence
        res = optimize.root_scalar(
            fun,
            bracket=bracket,
            args=args,
            method='brentq',
        )

        if not res.converged:
            logger.warning('root solver did not converge; flag: {}', res.flag)
        else:
            roots.append(res.root)

    return roots


def phenotype_ode(x, population):
    dxdt = (population['basal'] - x + population['alpha']
            * (x ** population['hill'] / (1 + x ** population['hill'])))
    return dxdt


def _root_solve_iter(coeffs):
    limits = (coeffs['basal'], coeffs['basal'] + coeffs['alpha'])
    return min(find_roots(phenotype_ode, limits, args=(coeffs,)))


def root_solver(population, executor=None):
    """Solve for lower steady state of ODE.

    This solver exploits that ODE has 1 or 3 distinct roots in the interval
    (basal, basal + alpha).
    """
    if executor is None:
        res_iter = map(_root_solve_iter, population)
    else:
        res_iter = executor.map(_root_solve_iter, population, chunksize=100)

    return np.fromiter(res_iter, dtype='float')


def phenotype(population, executor=None):
    """creates the phenotype"""

    population['basal'] = population['G'] + population['b'] * population['C']

    if np.any(population['basal'] <= 0):
        raise ValueError("population['basal'] is lower than or equal to zero.")

    if np.any(population['alpha'] <= 0):
        raise ValueError("population['alpha'] is lower than or equal to zero before mutation.")

    return root_solver(population, executor=executor)


def fitness(population):
    """calculates the fitness"""
    mean_phenotype = np.mean(population['phenotype'])
    # fitness_part_1: fitness component deriving from phenotypic (mis)match
    fitness_part_1 = (np.exp(-np.abs(population['E'] - population['phenotype']) * config['tau']))
    # fitness_part_2: fitness component deriving from disruptive selection (advantage of rare phenotypes)
    fitness_part_2 = (1 - np.exp(-np.abs(mean_phenotype - population['phenotype']) * config['phi']))
    W = (fitness_part_1 * fitness_part_2) - config['kd']
    W[W < 0.] = 0.
    return W


def mutation(population):
    """introduces random mutations for the genetic parameters"""
    n = len(population)

    mutation_coeffs = (
        # key, perturbation, lower bound, upper bound
        ('G', 0.05, 1e-9, 1),
        ('b', 0.05, 1e-9, 1),
        ('alpha', 0.05, 1, None),
        ('hill', 0.05, 1, None),
    )

    for key, perturbation, lower, upper in mutation_coeffs:
        mutation = np.where(
            np.random.random(n) <= config['mu'],
            np.random.normal(0, 0.05, n),
            0
        )

        population[key] = np.clip(
            population[key] + mutation,
            lower, upper
        )


def reproduction(population, executor=None):
    """produces next generation via mutation and natural selection"""
    mutation(population)
    population['phenotype'] = phenotype(population, executor=executor)
    if np.any(population['phenotype'] < 0):
        raise ValueError("population['phenotype'] is lower than zero.")

    population['fitness'] = fitness(population)
    if np.any(population['fitness'] < 0):
        raise ValueError("population['fitness'] is negative.")

    mean_fitness_before = np.mean(population['fitness'])
    if (mean_fitness_before == 0):
        raise ValueError("Mean fitness of population decreased to 0.")
    offspring = np.random.poisson(population['fitness'] / mean_fitness_before)

    logger.debug("Pop before reproduction: \n{}", population)

    population = np.repeat(population, offspring)
    if len(population) > config['n']:
        population = np.random.choice(population, config['n'], replace=False)
    elif len(population) < config['n']:
        clones = np.random.choice(np.arange(len(population)),
                                  config['n'] - len(population),
                                  replace=False)
        clone_rep = np.ones(len(population), dtype=np.int)
        clone_rep[clones] = 2
        population = np.repeat(population, clone_rep)

    return population


def main(nproc=1):
    logger.info("Configuration:\n{}\n", pprint.pformat(config))

    population = create_population(config['n'])

    list_environment = []
    list_C = []
    list_mean_fitness = []
    list_mean_b = []
    list_mean_phenotype = []

    with contextlib.ExitStack() as es:
        if nproc > 1:
            executor = es.enter_context(
                concurrent.futures.ProcessPoolExecutor(nproc)
            )
        else:
            executor = None

        for t in tqdm.tqdm(range(1, config['num_gen'] + 1)):
            population['E'], population['C'] = environment(t)
            population = reproduction(population, executor=executor)

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
        counts, bin_edges = np.histogram(population[var], bins=10)
        assert np.sum(counts) == config['n']
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bar_width = (80 * counts / config['n']).astype(np.int)

        hist_parts = ['', var, '#' * len(var)]
        for bin_center, bin_value, count in zip(bins, bar_width, counts):
            hist_parts.append(
                f'{bin_center:.2e} | {"█" * bin_value:<80} | ({count})'
            )
        hist_parts.append('')

        logger.info('\n'.join(hist_parts))

    return population


@click.command()
@click.option('-c', '--config', type=click.File('r'), default=None,
              help='Path to config file in TOML format')
@click.option('--nproc', help='Number of parallel processes for phenotype solver',
              default=multiprocessing.cpu_count(), type=int)
@click.option('--loglevel', default='info')
def cli(config, nproc, loglevel):
    logger.add(
        sys.stdout, colorize=sys.stdout.isatty(),
        format="<green>{time:%H:%M:%S}</green> <level>{message}</level>",
        level=loglevel.upper()
    )

    if config is not None:
        config.update(toml.load(config))

    return main(nproc=nproc)


if __name__ == '__main__':
    cli()

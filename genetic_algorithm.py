import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
import pickle
import random as rand
from copy import deepcopy

import numpy as np
import tensorflow as tf

from data_processing import sample_data
from helpers import exec_sca
from models import (build_small_cnn_ascad, load_small_cnn_ascad,
                    load_small_cnn_ascad_no_batch_norm)
from nn_genome import NeuralNetworkGenome


class GeneticAlgorithm:
    def __init__(self, max_gens, pop_size, mut_power, mut_rate, crossover_rate,
                 mut_power_decay_rate, truncation_proportion, atk_set_size,
                 parallelise=False, apply_fitness_inheritance=False,
                 elitism=False):
        self.max_gens = max_gens
        self.pop_size = pop_size
        self.mut_power = mut_power
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.mut_power_decay_rate = mut_power_decay_rate
        self.truncation_proportion = truncation_proportion
        self.atk_set_size = atk_set_size
        self.parallelise = parallelise
        self.apply_fi = apply_fitness_inheritance

        # Maintain the population and all offspring in self.population
        # The offspring occupy the second half of the array
        self.population = np.empty(pop_size*2, dtype=object)
        self.fitnesses = np.full(pop_size*2, 255, dtype=np.uint8)

        # Store useful information
        self.best_fitness_per_gen = np.empty(max_gens, dtype=np.uint8)

        # Parallelisation variables
        pool_size = round(min(self.pop_size*2, mp.cpu_count()*0.5))
        # pool_size = min(self.pop_size*2, len(os.sched_getaffinity(0)))
        if self.parallelise:
            self.pool = mp.Pool(pool_size)
        
        # Use a dictionary to enable simple selection method parametrisation
        selection_methods = {
            "roulette_wheel": self.roulette_wheel_selection,
            "tournament": self.tournament_selection
        }
    
    def __del__(self):
        if self.parallelise:
            self.pool.close()

    def run(self, nn, x_atk_full, y_atk_full, ptexts, true_subkey, subkey_i=2):
        """
        Runs the genetic algorithm with the parameters it was constructed with
        and returns the best found individual.
        """
        self.initialise_population(nn)

        # Track generational information
        gen = 0
        best_fitness = 256
        best_individual = None

        while gen < self.max_gens and best_fitness > 0:
            # Randomly sample the attack set
            (x_atk, y_atk) = \
                sample_data(self.atk_set_size, x_atk_full, y_atk_full)

            # Evaluate the fitness of each individual
            print("Evaluating fitness values...")
            self.evaluate_fitness(x_atk, y_atk, ptexts, true_subkey, subkey_i)
            if self.apply_fi:
                self.adjust_fitness()

            # Update the best known individual
            best_idx = np.argmin(self.fitnesses)
            best_fitness = self.fitnesses[best_idx]
            best_individual = self.population[best_idx]

            print("Selecting individuals...")
            # Rest of GA main loop, i.e. selection & offspring production
            # self.population[:self.pop_size] = self.roulette_wheel_selection()  # TODO: add truncation selection?
            self.population[:self.pop_size] = self.tournament_selection()
            print("Producing offspring...")
            self.population[self.pop_size:] = self.produce_offpsring()

            # Track useful information
            # TODO: Test best individual on a separate test to check generalisation
            self.best_fitness_per_gen[gen] = best_fitness
            print(f"Best fitness in generation {gen}: {best_fitness}")
            gen += 1
        
        return best_individual

    def initialise_population(self, nn):
        """
        Initialises a population of NNs with the given architecture parameters.
        """
        nn_weights = nn.get_weights()
        for i in range(len(self.population)):
            self.population[i] = NeuralNetworkGenome(nn_weights)
            self.population[i].random_weight_init()
            # TODO: maybe initialise weights randomly?
            # TODO: parallelise?

    def evaluate_fitness(self, x_atk, y_atk, ptexts, true_subkey, subkey_idx):
        """
        Computes and sets the current fitness value of each individual in the
        population and list of offspring.
        """
        if self.parallelise:
            # Set up a tuple of arguments for each concurrent process
            argss = [
                (self.population[i].weights, x_atk, y_atk, ptexts, true_subkey, subkey_idx)
                for i in range(len(self.population))
            ]
            # Run fitness evaluations in parallel
            # self.fitnesses = self.pool.starmap(exec_sca, argss)
            self.fitnesses = self.pool.starmap(evaluate_fitness, argss)

            # Update the individuals' fitness values
            for i in range(len(self.population)):
                self.population[i].fitness = self.fitnesses[i]
        else:
            # Run fitness evaluations sequentially
            for (i, indiv) in enumerate(self.population):
                self.fitnesses[i] = \
                    evaluate_fitness(indiv.weights, x_atk, y_atk, ptexts, true_subkey, subkey_idx)
                indiv.fitness = self.fitnesses[i]
    
    def adjust_fitness(self, fi_decay=0.2):
        """
        Adjusts each individual's fitness based on that of their parent(s) by
        applying fitness inheritance.

        Arguments:
            fi_decay: The fitness inheritance decay, which dictates the
            impact of the fitness of an individual's parent(s).
        """
        for (i, indiv) in enumerate(self.population):
            indiv.fitness = self.fitnesses[i] = round(
                (indiv.fitness + indiv.avg_parent_fitness * (1 - fi_decay))/2
            )

    def roulette_wheel_selection(self):
        """
        Selects potentially strong individuals by assigning each of them a
        selection probability that scales with their fitness. This is also
        known as fitness proportionate selection.

        This method assumes the fitness values have already been computed in
        the current generation.
        """
        new_population = np.empty(self.pop_size, dtype=object)

        # Invert fitnesses because fitness is minimised and min. fitness = 255
        inverted_fitnesses = np.fromiter(
            (255 - indiv.fitness for indiv in self.population),
            dtype=np.uint8
        )

        # Compute their sum so we can compute selection probabilities
        inv_fitness_sum = np.sum(inverted_fitnesses)

        # Compute individual fitness probabilities and sum them up cumulatively
        # to construct sections of the roulette wheel
        size = self.pop_size*2
        cumulative_select_probs = np.zeros(size, dtype=float)
        cumulative_prob = 0.0
        for i in range(size):
            cumulative_prob += inverted_fitnesses[i] / inv_fitness_sum
            cumulative_select_probs[i] = cumulative_prob

        # Build the new population by "spinning the wheel" repeatedly
        for i in range(self.pop_size):
            idx = cumulative_select_probs.searchsorted(np.random.uniform())
            new_population[i] = self.population[idx]

        return new_population

    def tournament_selection(self, t_size=3):
        """
        Selects potentially strong individuals by performing fitness-based
        tournaments with replacement.
        """
        new_population = np.empty(self.pop_size, dtype=object)
        for i in range(self.pop_size):
            # Pick a random participant, which starts as the winner by default
            winner = np.random.choice(self.population)

            # Compare (t_size - 1) more individuals based on their fitness 
            for j in range(t_size - 1):
                indiv = np.random.choice(self.population)
                if indiv.fitness < winner.fitness:
                    winner = indiv
            
            new_population[i] = winner
        
        return new_population
    
    def produce_offpsring(self):
        """
        Produces and returns offspring by applying either mutation or crossover
        on each individual, doubling the total population size.
        """
        offspring = np.empty(self.pop_size, dtype=object)
        for i in range(self.pop_size):
            parent0 = self.population[i]

            if np.random.uniform() < 0.5:
                parent1 = np.random.choice(self.population[:self.pop_size])
                offspring[i] = parent0.crossover(parent1, self.apply_fi)
            else:
                offspring[i] = parent0.mutate(self.mut_power, self.mut_rate, self.apply_fi)
        
        return offspring
    
    def save_results(self, best_indiv, experiment_name):
        """
        Saves the results, i.e. the best individual and the best fitnesses per
        generation, to a pickle file for later use.
        """
        with open(f"./results/{experiment_name}_ga_results.pickle", "wb") as f:
            ga_results = (best_indiv, self.best_fitness_per_gen)
            pickle.dump(ga_results, f)


def evaluate_fitness(weights, x_atk, y_atk, ptexts, true_subkey, subkey_idx):
    """
    Evaluates the fitness of an individual by using its weights to construct a
    new CNN, which is used to execute an SCA on the given data.

    Returns:
        The key rank obtained with the SCA.
    """
    cnn = load_small_cnn_ascad_no_batch_norm()
    cnn.set_weights(weights)
    return exec_sca(cnn, x_atk, y_atk, ptexts, true_subkey, subkey_idx)

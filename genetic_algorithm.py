import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
import pickle
import random as rand
from copy import deepcopy

import numpy as np
import tensorflow as tf

from data_processing import sample_data
from helpers import (exec_sca, compute_fitness, calc_max_fitness,
                     calc_min_fitness)
from metrics import MetricType
from models import (build_small_cnn_ascad, load_small_cnn_ascad,
                    load_small_cnn_ascad_no_batch_norm, load_small_mlp_ascad,
                    NN_LOAD_FUNC)
from nn_genome import NeuralNetworkGenome


class GeneticAlgorithm:
    def __init__(self, max_gens, pop_size, mut_power, mut_rate, crossover_rate,
                 mut_power_decay_rate, truncation_proportion, atk_set_size,
                 parallelise=False, apply_fitness_inheritance=False,
                 select_fun="roulette_wheel", metric_type=MetricType.KEYRANK,
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
        self.metric_type = metric_type

        # Maintain the population and all offspring in self.population
        # The offspring occupy the second half of the array
        self.population = np.empty(pop_size*2, dtype=object)

        # Precompute fitness-related variables
        self.max_fitness = max_base_f = calc_max_fitness(metric_type)
        max_unscaled_adj_fitness = adjust_fitness(max_base_f, max_base_f, 0.2)
        self.fitness_scaling = (1/max_unscaled_adj_fitness) * max_base_f
        self.min_fitness = calc_min_fitness(metric_type)

        # Store fitness-related information in arrays of the appropriate dtype
        dtype = np.uint8 if metric_type == MetricType.KEYRANK else np.float64
        self.fitness_dtype = dtype
        self.fitnesses = np.full(pop_size*2, self.max_fitness, dtype=dtype)
        self.best_fitness_per_gen = np.empty(max_gens, dtype=dtype)

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
        self.selection_method = selection_methods[select_fun]
    
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

        while gen < self.max_gens and best_fitness > self.min_fitness:
            # Randomly sample the attack set
            (x_atk, y_atk) = \
                sample_data(self.atk_set_size, x_atk_full, y_atk_full)

            # Evaluate the fitness of each individual
            print("Evaluating fitness values...")
            self.evaluate_fitness(x_atk, y_atk, ptexts, true_subkey, subkey_i)
            if self.apply_fi:
                self.adjust_fitnesses()

            # Update the best known individual
            best_idx = np.argmin(self.fitnesses)
            best_fitness = self.fitnesses[best_idx]
            best_individual = self.population[best_idx]

            print("Selecting individuals...")
            # Rest of GA main loop, i.e. selection & offspring production
            self.population[:self.pop_size] = self.selection_method()  # TODO: Add truncatiom selection?
            print("Producing offspring...")
            self.population[self.pop_size:] = self.produce_offpsring()

            # Track useful information
            # TODO: Test best individual on a separate test to check generalisation
            self.best_fitness_per_gen[gen] = best_fitness
            print(f"Best fitness in generation {gen}: {best_fitness}")

            self.mut_power *= self.mut_power_decay_rate
            gen += 1

        # Clean up
        for i in range(len(self.population)):
            self.fitnesses[i] = self.population[i].fitness
        if self.parallelise:
            self.pool.close()

        return best_individual

    def initialise_population(self, nn):
        """
        Initialises a population of NNs with the given architecture parameters.
        """
        weights = nn.get_weights()
        for i in range(len(self.population)):
            self.population[i] = NeuralNetworkGenome(weights, self.max_fitness)
            self.population[i].random_weight_init()
            # TODO: parallelise?

    def evaluate_fitness(self, x_atk, y_atk, ptexts, true_subkey, subkey_idx):
        """
        Computes and sets the current fitness value of each individual in the
        population and list of offspring.
        """
        if self.parallelise:
            # Set up a tuple of arguments for each concurrent process
            argss = [
                (self.population[i].weights, x_atk, y_atk, ptexts, true_subkey, subkey_idx, self.metric_type)
                for i in range(len(self.population))
            ]
            # Run fitness evaluations in parallel
            fitnesses = self.pool.starmap(evaluate_fitness, argss)

            # Update the individuals' fitness values
            for i in range(len(self.population)):
                self.population[i].fitness = self.fitnesses[i] = fitnesses[i]
        else:
            # Run fitness evaluations sequentially
            for (i, indiv) in enumerate(self.population):
                self.fitnesses[i] = \
                    evaluate_fitness(indiv.weights, x_atk, y_atk, ptexts, true_subkey, subkey_idx, self.metric_type)
                indiv.fitness = self.fitnesses[i]
    
    def adjust_fitnesses(self, fi_decay=0.2):
        """
        Adjusts each individual's fitness based on that of their parent(s) by
        applying fitness inheritance.

        Arguments:
            fi_decay: The fitness inheritance decay, which dictates the
            impact of the fitness of an individual's parent(s).
        """
        for (i, indiv) in enumerate(self.population):
            # indiv.fitness = self.fitnesses[i] = round(
            #     (indiv.fitness + indiv.avg_parent_fitness * (1 - fi_decay))/2
            # )
            f, fp = indiv.fitness, indiv.avg_parent_fitness

            fitness = adjust_fitness(f, fp, fi_decay, self.fitness_scaling)

            # Avoid rounding to 0 for (0.5 < f < 1.0) during integer conversion
            if self.metric_type == MetricType.KEYRANK:
                fitness = round(fitness)

            indiv.fitness = self.fitnesses[i] = fitness

    def roulette_wheel_selection(self):
        """
        Selects potentially strong individuals by assigning each of them a
        selection probability that scales with their fitness. This is also
        known as fitness proportionate selection.

        This method assumes the fitness values have already been computed in
        the current generation.
        """
        new_population = np.empty(self.pop_size, dtype=object)

        # Invert fitnesses because fitness is minimised
        inverted_fitnesses = np.fromiter(
            (self.max_fitness - indiv.fitness for indiv in self.population),
            dtype=self.fitness_dtype
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

    def get_results(self):
        """
        Returns the results, i.e. the best individual, the best fitnesses per
        generation, and a list of the top 10 individuals without saving them.
        """
        top_ten_indices = np.argsort(self.fitnesses)[:10]
        top_ten = self.population[top_ten_indices]
        best_indiv = top_ten[0]

        return (best_indiv, self.best_fitness_per_gen, top_ten)

    def save_results(self, best_indiv, experiment_name):
        """
        Saves the results, i.e. the best individual, the best fitnesses per
        generation, and a list of the top 10 individuals, to a pickle file for
        later use.
        """
        ga_results = None
        with open(f"results/{experiment_name}_ga_results.pickle", "wb") as f:
            top_ten_indices = np.argsort(self.fitnesses)[:10]
            top_ten = self.population[top_ten_indices]

            ga_results = (best_indiv, self.best_fitness_per_gen, top_ten)
            pickle.dump(ga_results, f)

        return ga_results

    @staticmethod
    def load_results(experiment_name):
        """
        Loads an experiment's results, i.e. the best individual, the best
        fitnesses per generation, and a list of the top 10 individuals, from a
        pickle file.
        """
        ga_results = None
        with open(f"results/{experiment_name}_ga_results.pickle", "rb") as f:
            ga_results = pickle.load(f)

        return ga_results


def evaluate_fitness(weights, x_atk, y_atk, ptexts, true_subkey, subkey_idx,
                     metric_type=MetricType.KEYRANK):
    """
    Evaluates the fitness of an individual by using its weights to construct a
    new CNN, which is used to execute an SCA on the given data.

    Returns:
        The key rank obtained with the SCA.
    """
    nn = NN_LOAD_FUNC()
    nn.set_weights(weights)

    return compute_fitness(nn, x_atk, y_atk, ptexts, metric_type, true_subkey, subkey_idx)
    # return exec_sca(cnn, x_atk, y_atk, ptexts, true_subkey, subkey_idx)


def adjust_fitness(fitness, avg_parent_fitness, fi_decay, scaling=1.0):
    """
    Adjusts and returns the given individual's fitness based on itself, its
    average parent fitness, and the given fitness inheritance decay value.
    """
    return scaling*(
        fitness + avg_parent_fitness * (1 - fi_decay)
    )

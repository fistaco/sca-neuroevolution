import multiprocessing as mp
# mp.set_start_method("spawn", force=True)
import pickle
import random as rand

import numpy as np
import tensorflow as tf

from data_processing import sample_traces
from helpers import (compute_fitness, calc_max_fitness, calc_min_fitness,
                     get_pool_size, ga_stagnation)
from metrics import MetricType
import models
from models import train
from nascty_param_limits import NasctyParamLimits
from cnn_genome import CnnGenome
from params import *
from plotting import plot_gens_vs_fitness


class NasctyCnnsGeneticAlgorithm:
    """
    Defines methods to run "Neuroevolution to Attack Side-Channel Traces
    Yielding Convolutional Neural Networks (NASCTY-CNNs), a genetic algorithm
    that evolves the parameters of CNNs for side-channel analysis.
    """

    def __init__(self, max_gens, pop_size, mut_rate, crossover_rate, atk_set_size,
                 parallelise=False, select_fun="tournament", t_size=3,
                 metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY,
                 truncation_proportion=1.0, n_atk_folds=1, remote=False):
        self.max_gens = max_gens
        self.pop_size = pop_size
        self.full_pop_size = pop_size*2  # Pop size when including offspring
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.truncation_proportion = truncation_proportion
        self.atk_set_size = atk_set_size
        self.n_atk_folds = n_atk_folds
        self.parallelise = parallelise
        self.metric_type = metric_type

        # Maintain the population and all offspring in self.population
        # The offspring occupy the second half of the array
        self.population = np.empty(pop_size*2, dtype=object)

        # Precompute fitness-related variables
        self.max_fitness = calc_max_fitness(metric_type)
        self.min_fitness = calc_min_fitness(metric_type)

        # Store fitness-related information in arrays of the appropriate dtype
        dtype = np.uint8 if metric_type == MetricType.KEYRANK else np.float64
        self.fitness_dtype = dtype
        self.fitnesses = np.full(pop_size*2, self.max_fitness, dtype=dtype)
        self.best_fitness_per_gen = np.empty(max_gens, dtype=dtype)

        if self.parallelise:
            pool_size = get_pool_size(remote, pop_size=pop_size)
            print(f"Initialising process pool with pool size = {pool_size}")

            self.pool = mp.Pool(pool_size)

        # Use a dictionary to enable simple selection method parametrisation
        selection_methods = {
            "roulette_wheel": self.roulette_wheel_selection,
            "tournament": self.tournament_selection,
            "unbiased_tournament": self.unbiased_tournament_selection
        }
        self.selection_method = selection_methods[select_fun]
        self.t_size = t_size

        # Define parameter limits
        self.param_limits = NasctyParamLimits()

    def __del__(self):
        if self.parallelise:
            self.pool.close()

    def run(self, x_train, y_train, pt_train, x_valid, y_valid, pt_valid, k,
            subkey_i=2, shuffle_traces=True, balanced=True, hw=False,
            static_seed=False):
        """
        Runs the genetic algorithm with the parameters it was constructed with
        and returns the best found individual.
        """
        # Ensure we're not attacking multiple unshuffled folds
        assert self.n_atk_folds == 1 or shuffle_traces, "Using static folds"

        # if self.metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        #     y_atk = tf.keras.utils.to_categorical(y_atk, (9 if hw else 256))

        self.initialise_population()

        # Track generational information
        gen = 0
        best_fitness = 7777777
        best_individual = None

        while gen < self.max_gens and best_fitness > self.min_fitness:
            seed = gen if self.n_atk_folds > 1 and not static_seed else 77
            np.random.seed(seed)

            # if self.metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
            #     y_atk = tf.keras.utils.to_categorical(y_atk,(9 if hw else 256))

            # Evaluate the fitness of each individual
            self.evaluate_fitness(
                x_train, y_train, pt_train, x_valid, y_valid, pt_valid, k,
                subkey_i, seed, shuffle_traces, balanced, hw
            )

            # Update the best known individual
            best_idx = np.argmin(self.fitnesses)
            best_fitness = self.fitnesses[best_idx]
            best_individual = self.population[best_idx]

            # Rest of GA main loop, i.e. selection & offspring production
            self.population[:self.pop_size] = self.selection_method()
            self.population[self.pop_size:] = self.produce_offpsring()

            # Track useful information
            self.best_fitness_per_gen[gen] = best_fitness
            print(f"Best fitness in generation {gen}: {best_fitness}")

            gen += 1

        # Clean up
        for i in range(len(self.population)):
            self.fitnesses[i] = self.population[i].fitness
        if self.parallelise:
            self.pool.close()

        return best_individual

    def initialise_population(self):
        """
        Initialises a population of CNNs with random parameters within limits
        defined in the given `NasctyParamLimits` object.
        """
        for i in range(len(self.population)):
            self.population[i] = CnnGenome.random(self.param_limits)

    def evaluate_fitness(self, x_train, y_train, pt_train, x_valid, y_valid,
                         pt_valid, true_subkey, subkey_idx, seed, shuffle=True,
                         balanced=True, hw=False):
        """
        Computes and sets the current fitness value of each individual in the
        population and list of offspring.
        """
        if self.parallelise:
            # Set up a tuple of arguments for each concurrent process
            argss = [
                (
                    self.population[i], x_train, y_train, pt_train,
                    x_valid, y_valid, pt_valid, true_subkey, subkey_idx,
                    self.metric_type, self.atk_set_size, self.n_atk_folds,
                    seed, shuffle, balanced, hw
                )
                for i in range(len(self.population))
            ]
            # Run fitness evaluations in parallel
            fitnesses = self.pool.starmap(multifold_fitness_eval, argss)

            # Update the individuals' fitness values
            for i in range(len(self.population)):
                self.population[i].fitness = self.fitnesses[i] = fitnesses[i]
        else:
            # Run fitness evaluations sequentially
            for (i, genome) in enumerate(self.population):
                genome.fitness = self.fitnesses[i] = multifold_fitness_eval(
                    genome, x_train, y_train, pt_train, x_valid, y_valid,
                    pt_valid, true_subkey, subkey_idx, self.metric_type,
                    self.atk_set_size, self.n_atk_folds, seed, shuffle,
                    balanced, hw
                )

    def roulette_wheel_selection(self, n_elites=1):
        """
        Selects potentially strong individuals by assigning each of them a
        selection probability that scales with their fitness. This is also
        known as fitness proportionate selection.

        This method assumes the fitness values have already been computed in
        the current generation.
        """
        new_population = np.empty(self.pop_size, dtype=object)

        # Potentially use a truncated population with the best fitnesses
        trunc_size = round(self.truncation_proportion*self.full_pop_size)
        pop = self.population[np.argsort(self.fitnesses)[:trunc_size]]

        # Invert fitnesses because fitness is minimised
        inverted_fitnesses = np.fromiter(
            (self.max_fitness - indiv.fitness for indiv in pop),
            dtype=self.fitness_dtype
        )

        # Compute their sum so we can compute selection probabilities
        inv_fitness_sum = np.sum(inverted_fitnesses)

        # Compute individual fitness probabilities and sum them up cumulatively
        # to construct sections of the roulette wheel
        cumulative_select_probs = np.zeros(trunc_size, dtype=float)
        cumulative_prob = 0.0
        for i in range(trunc_size):
            cumulative_prob += inverted_fitnesses[i] / inv_fitness_sum
            cumulative_select_probs[i] = cumulative_prob

        # Lock in the top n slots with the top n individuals
        top_n_idxs = np.argsort(self.fitnesses)[:n_elites]
        new_population[:n_elites] = self.population[top_n_idxs]

        # Build the new population by "spinning the wheel" repeatedly
        for i in range(n_elites, self.pop_size):
            idx = cumulative_select_probs.searchsorted(np.random.uniform())
            new_population[i] = pop[idx].clone()

        return new_population

    def tournament_selection(self, replace=False):
        """
        Selects potentially strong individuals by performing fitness-based
        tournaments.
        """
        # Potentially use a truncated population with the best fitnesses
        trunc_size = round(self.truncation_proportion*self.full_pop_size)
        trunc_idxs = np.argsort(self.fitnesses)[:trunc_size]
        pop, fits = self.population[trunc_idxs], self.fitnesses[trunc_idxs]

        new_population = np.empty(self.pop_size, dtype=object)
        for i in range(self.pop_size):
            # Pick 3 random participants and pick the fittest one
            idxs = np.random.choice(trunc_size, self.t_size, replace)
            new_population[i] = self.fitness_tournament(idxs, pop, fits)

        return new_population

    def unbiased_tournament_selection(self, t_size=4):
        """
        Selects potentially strong individuals by performing fitness-based
        tournaments where each individual participates in exactly `(t_size/2)`
        tournaments.
        """
        assert t_size % 2 == 0, "t_size cannot be odd for unbiased t_select"

        # Potentially use a truncated population with the best fitnesses
        trunc_size = round(self.truncation_proportion*self.full_pop_size)
        trunc_idxs = np.argsort(self.fitnesses)[:trunc_size]
        pop, fits = self.population[trunc_idxs], self.fitnesses[trunc_idxs]

        # Set up permutations of the possibly truncated population
        idx_permuts = np.array([
            np.random.permutation(trunc_size) for _ in range(t_size//2)
        ], dtype=np.uint16)

        # Construct tournament groups by lining up halves of the permutations
        tourn_permuts = np.zeros((t_size, self.pop_size), dtype=np.uint16)
        for i in range(t_size//2):
            tourn_permuts[i] = idx_permuts[i][:self.pop_size]
            tourn_permuts[i + 1] = idx_permuts[i][self.pop_size:]

        # Select members for the population with `pop_size` tournaments
        new_population = np.empty(self.pop_size, dtype=object)
        for i in range(self.pop_size):
            idxs = tourn_permuts[:, i]
            new_population[i] = self.fitness_tournament(idxs, pop, fits)

        return new_population

    def fitness_tournament(self, idxs, population, fitnesses):
        """
        Executes single tournament for tournament selection with individuals
        corresponding to the given indices and returns the winner (cloned, not
        by reference).
        """
        indivs = population[idxs]
        winner = indivs[np.argmin(fitnesses[idxs])]

        return winner.clone()

    def produce_offpsring(self):
        """
        Produces and returns offspring by applying either mutation or crossover
        on each individual, doubling the total population size.
        """
        offspring = np.empty(self.pop_size, dtype=object)
        for i in range(self.pop_size):
            parent0 = self.population[i]

            if np.random.uniform() < self.crossover_rate:
                parent1 = np.random.choice(self.population[:self.pop_size])
                offspring[i] = parent0.crossover(parent1)
            else:
                offspring[i] = parent0.mutate()
        
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


def evaluate_nascty_fitness(genome, x_train, y_train, pt_train, x_valid,
                            y_valid, pt_valid, true_subkey, subkey_idx,
                            metric_type, n_folds, seed, shuffle=True,
                            balanced=True, hw=False):
    """
    Evaluates the fitness of the given `genome` by using its parameter
    specifications to construct a CNN, which is then trained with
    backpropagation on the given training data. The fitness of this CNN is then
    computed on the given validation data according to the given `metric_type`.

    Returns:
        An `np.float64` or `int` object representing the `genome`'s fitness
        according to the given `metric_type`.
    """
    np.random.seed(seed)

    n_classes = 9 if hw else 256
    if metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        y_train = tf.keras.utils.to_categorical(y_train, n_classes)
        y_valid = tf.keras.utils.to_categorical(y_valid, n_classes)

    nn = train(genome.phenotype(), x_train, y_train, epochs=50)

    return compute_fitness(
        nn, x_valid, y_valid, pt_valid, metric_type, true_subkey, len(x_valid),
        subkey_idx, hw, preds=None, n_folds=n_folds
    )


def multifold_fitness_eval(weights, x_atk, y_atk, pt_atk, true_subkey,
                           subkey_idx, metric_type, atk_set_size, n_folds,
                           seed, shuffle=True, balanced=True, hw=False):
    """
    Evaluates the fitness of an individual by using its weights to construct a
    new NN, which is used to execute an SCA on the given data.

    Returns:
        The key rank obtained with the SCA.
    """
    np.random.seed(seed)  # Ensure each indiv is evaluated on the same folds

    nn = models.NN_LOAD_FUNC(*models.NN_LOAD_ARGS)
    nn.set_weights(weights)

    n_classes = 9 if hw else 256

    x, y, pt = sample_traces(
        atk_set_size, x_atk, y_atk, pt_atk, n_classes, shuffle=shuffle,
        balanced=balanced
    )

    if metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        y = tf.keras.utils.to_categorical(y, (9 if hw else 256))

    return compute_fitness(
        nn, x, y, pt, metric_type, true_subkey, atk_set_size, subkey_idx, hw,
        preds=None, n_folds=n_folds
    )


def sgd_train(weights, x_train, y_train, seed, epochs=5):
    """
    Trains the given NN with a default scheme for a given amount of epochs and
    returns the resulting weights.
    """
    np.random.seed(seed)  # Ensure each indiv is trained on the same fold

    nn = models.NN_LOAD_FUNC(*models.NN_LOAD_ARGS)
    nn.set_weights(weights)

    y_cat = tf.keras.utils.to_categorical(y_train)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    nn.compile(optimizer, loss_fn)
    _ = nn.fit(x_train, y_cat, batch_size=50, epochs=epochs, verbose=0)

    return nn.get_weights()


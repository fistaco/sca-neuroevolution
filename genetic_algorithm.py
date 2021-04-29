import multiprocessing as mp
# mp.set_start_method("spawn", force=True)
import pickle
import random as rand
from copy import deepcopy

import numpy as np
import tensorflow as tf

from data_processing import sample_data, balanced_sample, sample_traces
from helpers import (exec_sca, compute_fitness, calc_max_fitness,
                     calc_min_fitness, get_pool_size)
from metrics import MetricType
import models
from models import (build_small_cnn_ascad, load_small_cnn_ascad,
                    load_small_cnn_ascad_no_batch_norm, load_small_mlp_ascad)
from nn_genome import NeuralNetworkGenome
from params import *
from plotting import plot_gens_vs_fitness


class GeneticAlgorithm:
    def __init__(self, max_gens, pop_size, mut_power, mut_rate, crossover_rate,
                 mut_power_decay_rate, truncation_proportion, atk_set_size,
                 parallelise=False, apply_fitness_inheritance=False,
                 select_fun="tournament", metric_type=MetricType.KEYRANK,
                 n_atk_folds=1, remote=False, t_size=3):
        self.max_gens = max_gens
        self.pop_size = pop_size
        self.full_pop_size = pop_size*2  # Pop size when including offspring
        self.mut_power = mut_power
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.mut_power_decay_rate = mut_power_decay_rate
        self.truncation_proportion = truncation_proportion
        self.atk_set_size = atk_set_size
        self.n_atk_folds = n_atk_folds
        self.parallelise = parallelise
        self.apply_fi = apply_fitness_inheritance
        self.metric_type = metric_type

        # Maintain the population and all offspring in self.population
        # The offspring occupy the second half of the array
        self.population = np.empty(pop_size*2, dtype=object)

        # Precompute fitness-related variables
        self.max_fitness = max_base_f = calc_max_fitness(metric_type)
        max_unscaled_adj_fitness = adjust_fitness(max_base_f, max_base_f, 0.2)
        # self.fitness_scaling = (1/max_unscaled_adj_fitness) * max_base_f
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

    def __del__(self):
        if self.parallelise:
            self.pool.close()

    def run(self, nn, x_atk_full, y_atk_full, ptexts, true_subkey, subkey_i=2,
            shuffle_traces=True, balanced=True, debug=False):
        """
        Runs the genetic algorithm with the parameters it was constructed with
        and returns the best found individual.
        """
        # Ensure we're not attacking multiple unshuffled folds
        assert self.n_atk_folds == 1 or shuffle_traces, "Using static folds"

        # if self.metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        #     y_atk_full = tf.keras.utils.to_categorical(y_atk_full, 256)

        self.initialise_population(nn)

        # Track generational information
        gen = 0
        best_fitness = 256
        best_individual = None

        while gen < self.max_gens and best_fitness > self.min_fitness:
            # Obtain balanced, optionally random samples from the attack set
            atk_sets = [
                sample_traces(self.atk_set_size, x_atk_full, y_atk_full,
                                ptexts, shuffle=shuffle_traces,
                                balanced=balanced)
                for _ in range(self.n_atk_folds)
            ]

            # Note: 5k profiling traces may be insufficient
            # Note: try computing fitness over 100 folds
            if self.metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
                y_atk = tf.keras.utils.to_categorical(y_atk, 256)

            # Evaluate the fitness of each individual
            self.evaluate_fitness(atk_sets, true_subkey, subkey_i)
            if self.apply_fi:
                self.adjust_fitnesses()

            # Update the best known individual
            best_idx = np.argmin(self.fitnesses)
            best_fitness = self.fitnesses[best_idx]
            best_individual = self.population[best_idx]

            # Rest of GA main loop, i.e. selection & offspring production
            self.population[:self.pop_size] = self.selection_method()  # TODO: Add truncation selection?
            self.population[self.pop_size:] = self.produce_offpsring()

            # Track useful information
            # TODO: Test best individual on a separate test set to observe generalisation
            self.best_fitness_per_gen[gen] = best_fitness
            print(f"Best fitness in generation {gen}: {best_fitness}")

            self.mut_power *= self.mut_power_decay_rate
            gen += 1

            if debug:
                for indiv in self.population:
                    print(f"({indiv.indiv_id}, {indiv.fitness})")

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
            self.population[i] = NeuralNetworkGenome(
                weights, self.max_fitness, i
            )
            self.population[i].random_weight_init()

    def evaluate_fitness(self, atk_sets, true_subkey, subkey_idx):
        """
        Computes and sets the current fitness value of each individual in the
        population and list of offspring.
        """
        if self.parallelise:
            # Set up a tuple of arguments for each concurrent process
            argss = [
                (
                    self.population[i].weights, atk_sets, true_subkey,
                    subkey_idx, self.metric_type, self.atk_set_size
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
            for (i, indiv) in enumerate(self.population):
                indiv.fitness = self.fitnesses[i] = multifold_fitness_eval(
                    indiv.weights, atk_sets, true_subkey, subkey_idx,
                    self.metric_type, self.atk_set_size
                )

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
            f = indiv.fitness
            fp = indiv.avg_parent_fitness or f

            fitness = adjust_fitness(f, fp, fi_decay)

            # Avoid rounding to 0 for (0.5 < f < 1.0) during integer conversion
            if self.metric_type == MetricType.KEYRANK:
                fitness = round(fitness)

            indiv.fitness = self.fitnesses[i] = fitness
            if not indiv.avg_parent_fitness:
                indiv.avg_parent_fitness = fitness

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
        tournaments with replacement.
        """
        new_population = np.empty(self.pop_size, dtype=object)
        for i in range(self.pop_size):
            # Pick 3 random participants and pick the fittest one
            idxs = np.random.choice(self.full_pop_size, self.t_size, replace)
            new_population[i] = self.fitness_tournament(idxs)

        return new_population

    def unbiased_tournament_selection(self, t_size=4):
        """
        Selects potentially strong individuals by performing fitness-based
        tournaments where each individual participates in exactly `(t_size/2)`
        tournaments.
        """
        assert t_size % 2 == 0, "t_size cannot be odd for unbiased t_select"

        # Set up permutations of the full population, including offspring
        idx_permuts = np.array([
            np.random.permutation(self.full_pop_size) for _ in range(t_size//2)
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
            new_population[i] = self.fitness_tournament(idxs)

        return new_population

    def fitness_tournament(self, idxs):
        """
        Executes single tournament for tournament selection with individuals
        corresponding to the given indices and returns the winner (cloned, not
        by reference).
        """
        indivs = self.population[idxs]
        winner = indivs[np.argmin(self.fitnesses[idxs])]

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
                     metric_type, atk_set_size):
    """
    Evaluates the fitness of an individual by using its weights to construct a
    new NN, which is used to execute an SCA on the given data.

    Returns:
        The key rank obtained with the SCA.
    """
    nn = models.NN_LOAD_FUNC()
    nn.set_weights(weights)

    return compute_fitness(
        nn, x_atk, y_atk, ptexts, metric_type, true_subkey, atk_set_size,
        subkey_idx
    )


def multifold_fitness_eval(weights, atk_sets, true_subkey, subkey_idx,
                           metric_type, atk_set_size):
    """
    Evaluates the fitness of an individual by using its weights to construct a
    new NN, which is used to execute an SCA on the given data.

    Returns:
        The key rank obtained with the SCA.
    """
    nn = models.NN_LOAD_FUNC()
    nn.set_weights(weights)

    return np.mean([
        compute_fitness(
            nn, atk_set[0], atk_set[1], atk_set[2], metric_type, true_subkey,
            atk_set_size, subkey_idx
        )
        for atk_set in atk_sets
    ])


def adjust_fitness(fitness, avg_parent_fitness, fi_decay, scaling=0.5):
    """
    Adjusts and returns the given individual's fitness based on itself, its
    average parent fitness, and the given fitness inheritance decay value.
    """
    # res = scaling*(
    #     fitness + avg_parent_fitness * (1 - fi_decay)
    # )
    # print(f"f={fitness}, fp={avg_parent_fitness}, f_adj={res}")
    return scaling*(
        fitness + avg_parent_fitness * (1 - fi_decay)
    )


def train_nn_with_ga(
        nn,
        x_train,
        y_train,
        pt_train,
        k_train,
        subkey_idx,
        max_gens=MAX_GENERATIONS,
        pop_size=POPULATION_SIZE,
        mut_power=MUTATION_POWER,
        mut_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        mut_power_decay_rate=MUTATION_POWER_DECAY,
        truncation_proportion=TRUNCATION_PROPORTION,
        atk_set_size=ATTACK_SET_SIZE,
        parallelise=False,
        apply_fi=False,
        select_fn=SELECTION_FUNCTION,
        t_size=3,
        metric_type=METRIC_TYPE,
        shuffle_traces=True,
        n_atk_folds=1,
        remote=False,
        plot_fit_progress=True,
        exp_name="",
        debug=False
    ):
    """
    Trains the weights of the given NN on the given data set by running it
    through the genetic algorithm with  the given parameters.

    Returns:
        The trained NN.
    """
    weights = nn.get_weights()
    ga = GeneticAlgorithm(
        max_gens,
        pop_size,
        mut_power,
        mut_rate,
        crossover_rate,
        mut_power_decay_rate,
        truncation_proportion,
        atk_set_size,
        parallelise,
        apply_fi,
        select_fn,
        metric_type,
        n_atk_folds=n_atk_folds,
        remote=remote,
        t_size=t_size
    )

    # Obtain the best network resulting from the GA
    best_indiv = ga.run(
        nn, x_train, y_train, pt_train, k_train, subkey_idx,
        shuffle_traces=shuffle_traces, debug=debug
    )
    nn.set_weights(best_indiv.weights)

    if plot_fit_progress and exp_name:
        plot_gens_vs_fitness(exp_name, ga.best_fitness_per_gen)

    return nn

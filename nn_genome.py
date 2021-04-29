import numpy as np
from copy import deepcopy

from helpers import exec_sca
from metrics import MetricType


class NeuralNetworkGenome:
    def __init__(self, init_weights, init_parent_fitness=255, indiv_id=-1):
        # self.model = model
        self.weights = deepcopy(init_weights)  # List of numpy arrays
        self.fitness = -1
        self.indiv_id = indiv_id
        # TODO: multiple ways of initialising weights?

        self.avg_parent_fitness = None

    def random_weight_init(self):
        """
        Sets each weight to a random value in the range [-1, 1).
        """
        for i in range(len(self.weights)):  # Iterate over layers
            for j in range(self.weights[i].shape[-1]):  # Iterate over weights
                self.weights[i][..., j] = np.float32(np.random.uniform(-1, 1))

    def mutate(self, mut_power, mut_rate, apply_fitness_inheritance=False,
               fi_decay=0.2):
        """
        Creates a new child and mutates each of its weights by adding a number
        from the range [-mut_power, mut_power] if the mut_rate check passes for
        that weight. Returns the resulting child.
        """
        child = self.clone()

        for i in range(len(self.weights)):  # Iterate over layers
            for j in range(self.weights[i].shape[-1]):  # Iterate over weights
                if np.random.uniform() < mut_rate:
                    mut_val = np.random.uniform() * 2 * mut_power - mut_power
                    child.weights[i][..., j] += mut_val

        if apply_fitness_inheritance:
            child.avg_parent_fitness = self.fitness
            self.avg_parent_fitness = self.fitness

        return child

    def crossover(self, other, apply_fitness_inheritance=False, fi_decay=0.2):
        """
        Applies crossover by uniformly selecting weights from both parents and
        returns the resulting child.
        """
        child = self.clone()

        for i in range(len(self.weights)):  # Iterate over layers
            for j in range(self.weights[i].shape[-1]):  # Iterate over weights
                other_w = other.weights[i][..., j]

                # Replace with uniform probability
                if np.random.uniform() < 0.5:
                    child.weights[i][..., j] = other_w

        if apply_fitness_inheritance:
            child.avg_parent_fitness = (self.fitness + other.fitness)/2
            self.avg_parent_fitness = self.fitness

        return child

    def evaluate_fitness(self, x_atk, y_atk, ptexts, true_subkey):
        """
        Evaluates, sets, and returns this individual's fitness by predicting
        labels for a given data set and computing the key rank, which serves as
        a fitness value.

        Arguments:
            x_atk: Data set of power traces that is assumed to already be
            reshaped for compatibility with CNNs if necessary.

            y_atk: The labels corresponding to the given traces, which can
            represent different things depending on the data set in question.

            ptexts: The plaintexts corresponding to the traces, given as arrays
            of 8-bit integers.

            true_subkey: The actual subkey used at the targeted subkey index.
        """
        # Use the model to perform an SCA and obtain the key rank
        key_rank = exec_sca(self.model, x_atk, y_atk, ptexts, true_subkey)

        self.fitness = key_rank  # Doesnt't work if multiprocessing is used
        return key_rank

    def clone(self):
        """
        Returns a deep copy of this genome.
        """
        # clone = NeuralNetworkGenome(keras.models.clone_model(self.model))
        # clone.weights = deepcopy(self.weights)
        # clone.model.set_weights(clone.weights)
        # clone.fitness = self.fitness
        clone = NeuralNetworkGenome(
            self.weights, self.avg_parent_fitness, self.indiv_id
        )
        clone.fitness = self.fitness

        return clone

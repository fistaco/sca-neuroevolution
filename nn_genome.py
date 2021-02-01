import numpy as np
from copy import deepcopy
from tensorflow import keras

from helpers import exec_sca


class NeuralNetworkGenome:
    def __init__(self, model):
        self.model = model
        self.weights = model.get_weights()  # List of numpy arrays
        self.fitness = -1  # TODO: evaluate fitness upon creation?
        # TODO: multiple ways of initialising weights?

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

        # Manually reconfigure model weights
        self.model.set_weights(self.weights)

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

        # Manually reconfigure model weights
        self.model.set_weights(self.weights)

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

        self.fitness = key_rank
        return key_rank

    def clone(self):
        """
        Returns a deep copy of this genome.
        """
        clone = NeuralNetworkGenome(keras.models.clone_model(self.model))
        clone.weights = deepcopy(self.weights)
        clone.model.set_weights(clone.weights)
        clone.fitness = self.fitness

        return clone
from copy import deepcopy
from enum import Enum

from keras import Input, Model
from keras.layers import Conv1D, Dense, Flatten
import numpy as np
from numpy.random import randint


class CnnGenome:
    """
    A genome encoding of convolutional neural networks used in the NASCTY-CNNs
    genetic algorithm.
    """
    def __init__(self):
        self.conv_blocks = []
        self.dense_layers = []

        # If there are no conv blocks, apply a possible pooling layer
        self.pool_before_dense = False

    @staticmethod
    def random(limits):
        """
        Creates a random CnnGenome with parameters randomly set within a range
        defined in the given `NasctyParamLimits` object.
        """
        genome = CnnGenome()

        for _ in range(limits.n_conv_blocks_min, limits.n_conv_blocks_max + 1):
            genome.dense_layers.append(ConvBlockGene.random(limits))
        for _ in range(limits.n_dense_layers_min, limits.n_dense_layers_max + 1):
            genome.dense_layers.append(ConvBlockGene.random(limits))

        if len(genome.conv_blocks) == 0:
            genome.pool_before_dense = bool(randint(2))

        return genome

    def mutate(self, param_limits):
        """
        Mutates this genome by uniformly randomly adding a layer,
        removing a layer, or modifying all numerical parameters with
        probability 1/`n` where `n` is the total number of mutable parameters.
        """
        mut_type = randint(3)
        if mut_type == 0:
            self.add_random_layer(param_limits)
        elif mut_type == 1:
            self.remove_random_layer(param_limits)
        elif mut_type == 2:
            self.modify_parameters(param_limits)

    def crossover(self, other):
        """
        Constructs a child through crossover with this genome and the given
        `other` genome. This is achieved by TODO.
        """
        pass

    def add_random_layer(self, limits):
        """
        Adds a randomly initialised convolution block or dense layer if the
        layer limit allows it.
        """
        if randint(2) == 0:
            if len(self.conv_blocks) < limits.n_conv_blocks_max:
                self.conv_blocks.append(ConvBlockGene.random())
        elif len(self.dense_layers) < limits.n_dense_layers_max:
            self.dense_layers.append(DenseLayerGene.random())

    def remove_random_layer(self, limits):
        """
        Removes a random convolution block or dense layer if the layer limit
        allows it.
        """
        if randint(2) == 0:
            if len(self.conv_blocks) > limits.n_conv_blocks_min:
                del self.conv_blocks[randint(len(self.conv_blocks))]
        elif len(self.dense_layers) > limits.n_dense_layers_min:
            del self.dense_layers[randint(len(self.dense_layers))]

    def modify_parameters(self, limits):
        """
        Modifies all genome parameters with equal probability within an
        interval specific to each parameter.
        """
        n = len(self.conv_blocks)*6 + len(self.dense_layers)*1
        if (len(self.conv_blocks)) == 0:
            n += 1  # To modify self.pool_before_dense
            self.pool_before_dense = apply_boolean_mutation_with_prob(
                self.pool_before_dense, 1/n)
        mut_prob = 1/n

        for conv_block in self.conv_blocks:
            conv_block.mutate(mut_prob, limits)
        for dense_layer in self.dense_layers:
            dense_layer.mutate(mut_prob, limits)


    def phenotype(self):
        """
        Constructs and returns the phenotype corresponding to this genome, i.e.
        its expression as a Keras CNN model.
        """
        pass

    def clone(self):
        """
        Returns a duplicate clone (by value) of this genome.
        """
        clone = CnnGenome()
        clone.conv_blocks = deepcopy(self.conv_blocks)
        clone.dense_layers = deepcopy(self.dense_layers)

        return clone


class PoolType(Enum):
    """
    The type of pooling layer used for some neural network.
    """
    AVERAGE = 0
    MAX = 1

    def mutate_with_prob(self, mut_prob):
        """
        Returns AVERAGE if the current pool type is MAX, and returns MAX if the
        current pool type is AVERAGE.
        """
        return PoolType(not self.value) if np.random.uniform() < mut_prob \
            else self


class ConvBlockGene:
    """
    A convolutional block gene describing a convolutional filter, an optional
    batch normalisation layer, and a pooling layer.
    """
    def __init__(self, n_filters=2, filter_size=1, batch_norm=True,
                 pool_type=PoolType.AVERAGE, pool_size=2, pool_stride=2):
        self.n_filters = n_filters
        self.filter_size = filter_size

        self.batch_norm = batch_norm

        self.pool_type = pool_type
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    @staticmethod
    def random(limits):
        """
        Creates a random ConvBlockGene with parameters randomly set within a
        range defined in the given `NasctyParamLimits` object.
        """
        return ConvBlockGene(
            n_filters=randint(limits.n_filters_min, limits.n_filters_max + 1),
            filter_size=randint(limits.filter_size_min, limits.filter_size_max + 1),
            batch_norm=randint(2),
            pool_size=randint(limits.pool_size_min, limits.pool_size_max + 1),
            pool_stride=randint(limits.pool_stride_min, limits.pool_stride_max + 1)
        )

    def mutate(self, mut_prob, limits):
        """
        Mutates all of this gene's parameters with probability `mut_prob`
        within limits specific to each parameter.
        """
        self.n_filters = apply_polynom_mutation_with_prob(
            self.n_filters, limits.n_filters_min, limits.n_filters_max, mut_prob)
        self.filter_size = apply_polynom_mutation_with_prob(
            self.filter_size, limits.filter_size_min, limits.filter_size_max, mut_prob)

        self.batch_norm = apply_boolean_mutation_with_prob(self.batch_norm, mut_prob)

        self.pool_type = self.pool_type.mutate_with_prob(mut_prob)
        self.pool_size = apply_polynom_mutation_with_prob(
            self.pool_size, limits.pool_size_min, limits.pool_size_max, mut_prob)
        self.pool_stride = apply_polynom_mutation_with_prob(
            self.pool_stride, limits.pool_stride_min, limits.pool_stride_max, mut_prob)

    def clone(self):
        """
        Returns a duplicate clone (by value) of this gene.
        """
        clone = ConvBlockGene(
            self.n_filters, self.filter_size, self.batch_norm, self.pool_type,
            self.pool_size, self.pool_stride
        )
        return clone


class DenseLayerGene:
    """
    A dense layer gene describing the parameters of a fully-connected neural
    network layer.
    """
    def __init__(self, n_neurons=1, act_func="selu", weight_init="he_uniform"):
        self.n_neurons = n_neurons

        self.act_func = act_func
        self.weight_init = weight_init

    @staticmethod
    def random(limits):
        """
        Creates a random DenseLayerGene with parameters randomly set within a
        range defined in the given `NasctyParamLimits` object.
        """
        n_neurons = randint(
            limits.n_dense_neurons_min, limits.n_dense_neurons_max + 1
        )

        return DenseLayerGene(n_neurons)

    def mutate(self, mut_prob, limits):
        """
        Mutates all of this gene's parameters with probability `mut_prob`
        within limits specific to each parameter.
        """
        self.n_neurons = apply_polynom_mutation_with_prob(
            self.n_neurons, limits.n_dense_neurons_min,
            limits.n_dense_neurons_max, mut_prob
        )

    def clone(self):
        """
        Returns a duplicate clone (by value) of this gene.
        """
        clone = DenseLayerGene(self.n_neurons)
        return clone


def apply_polynomial_mutation(x, lo, hi, eta):
    """
    Applies polynomial mutation to the given numerical variable `x` according
    to the given boundaries `lo` and `hi` and returns the result.

    Polynomial mutation parameter `eta` determines the range in which a
    variable may be mutated, with higher eta resulting in a smaller mutation
    range.
    """
    u = np.random.uniform()
    if u <= 0.5:
        delta_l = (2*u)**(1/(1 + eta)) - 1
        return x + delta_l * (x - lo)
    else:
        delta_r = 1 - (2*(1 - u))**(1/(1 + eta))
        return x + delta_r * (hi - x)


def apply_polynom_mutation_with_prob(x, lo, hi, eta, mut_prob):
    """
    Applies polynomial mutation to the given numerical variable `x` according
    to the given boundaries `lo` and `hi` with probability `mut_prob`.
    """
    if np.random.uniform() < mut_prob:
        apply_polynomial_mutation(x, lo, hi, eta)


def apply_boolean_mutation_with_prob(b, mut_prob):
    """
    Sets bool `b` to the opposite of its current boolean value with probability
    `mut_prob`.
    """
    if np.random.uniform() < mut_prob:
        return not b

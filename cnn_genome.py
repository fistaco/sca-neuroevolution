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
        self.pool_before_dense = None

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
            # TODO: Modify all parameters with equal probability within an interval
            # specific to each parameter.
            pass

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

    def clone(self):
        """
        Returns a duplicate clone (by value) of this gene.
        """
        clone = DenseLayerGene(self.n_neurons)
        return clone


def apply_polynomial_mutation(x, lo, hi):
    """
    Applies polynomial mutation to the given numerical variable `x` according
    to the given extremum boundaries `lo` and `hi` and returns the result.
    """
    pass

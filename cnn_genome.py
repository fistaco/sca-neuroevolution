from copy import deepcopy
from enum import Enum

from keras import Input, Model
from keras.layers import AveragePooling1D, MaxPooling1D, Conv1D, BatchNormalization, Dense, Flatten
import numpy as np
from numpy.random import randint

from nascty_enums import CrossoverType, PoolType


class CnnGenome:
    """
    A genome encoding of convolutional neural networks used in the NASCTY-CNNs
    genetic algorithm.
    """
    def __init__(self):
        self.conv_blocks = []
        self.dense_layers = []

        # A pooling layer is always encoded, but is only mutated or expressed
        # in the phenotype if there are no conv blocks.
        self.pool_before_dense = PoolingGene()

        self.fitness = 777

    @staticmethod
    def random(limits):
        """
        Creates a random CnnGenome with parameters randomly set within a range
        defined in the given `NasctyParamLimits` object.
        """
        genome = CnnGenome()

        for _ in range(randint(limits.n_conv_blocks_min, limits.n_conv_blocks_max + 1)):
            genome.conv_blocks.append(ConvBlockGene.random(limits))
        for _ in range(randint(limits.n_dense_layers_min, limits.n_dense_layers_max + 1)):
            genome.dense_layers.append(DenseLayerGene.random(limits))

        genome.pool_before_dense = PoolingGene.random(limits)

        return genome

    def mutate(self, param_limits, polynom_mutation_eta=20):
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
            self.modify_parameters(param_limits, polynom_mutation_eta)

    def crossover(self, other, crossover_type):
        """
        Constructs a child through crossover with this genome and the given
        `other` genome according to the given `crossover_type`.
        """
        if crossover_type == CrossoverType.ONEPOINT:
            return self.onepoint_crossover(other)
        elif crossover_type == CrossoverType.PARAMETERWISE:
            return self.parameterwise_crossover(other)

    def onepoint_crossover(self, other):
        """
        Constructs two offspring genomes through one-point crossover separately
        within the lists of convolution blocks and dense layers.
        """
        c0, c1 = CnnGenome(), CnnGenome()

        c0.conv_blocks, c1.conv_blocks = \
            self.op_layer_crossover(self.conv_blocks, other.conv_blocks)
        c0.dense_layers, c1.dense_layers = \
            self.op_layer_crossover(self.dense_layers, other.dense_layers)

        c0.pool_before_dense, c1.pool_before_dense = \
            randomise_order(self.pool_before_dense, other.pool_before_dense)

        return (c0, c1)

    def op_layer_crossover(self, p0_layers, p1_layers):
        """
        Returns two lists of layers constructed through one-point crossover on
        two parents' respective layer lists.
        """
        p0_cutoff = self.onepoint_crossover_cutoff(p0_layers)
        p1_cutoff = self.onepoint_crossover_cutoff(p1_layers)

        # Take the cutoff and limit the total number at 5 layers
        c0_layers = (p0_layers[:p0_cutoff] + p1_layers[p1_cutoff:])[:5]
        c1_layers = (p1_layers[:p1_cutoff] + p0_layers[p0_cutoff:])[:5]

        return (c0_layers, c1_layers)

    def onepoint_crossover_cutoff(self, layers):
        """
        Finds a random cutoff point for one-point crossover in the given
        `layers` list.
        """
        if len(layers) == 0:
            return 0

        return randint(len(layers) + 1)  # Add 1 so we can cut at the end too

    def parameterwise_crossover(self, other):
        """
        Constructs two offspring genomes through parameterwise crossover
        between this genome and the `other` genome, where each parameter for
        the first offspring is randomly taken from one of the parents, and the
        same parameter for the other offspring is taken from the other.

        This is achieved by first lining up each gene of the same type for
        crossover. When the number of convolution or dense genes does not
        match up, the remaining layers are appended to one of the offspring
        unmodified.
        """
        c0, c1 = CnnGenome(), CnnGenome()

        c0.conv_blocks, c1.conv_blocks = self.paramwise_co_on_layers(
            self.conv_blocks, other.conv_blocks)
        c0.dense_layers, c1.dense_layers = self.paramwise_co_on_layers(
            self.dense_layers, other.dense_layers)

        self.pool_before_dense, other.pool_before_dense = \
            self.pool_before_dense.param_crossover(other.pool_before_dense)

        return (c0, c1)

    def paramwise_co_on_layers(self, p0_layers, p1_layers):
        """
        Creates two offspring lists by performing parameterwise crossover on
        `p0_layers` and `p1_layers`, which are lists containing layers of the
        same type.
        """
        c0_layers = []
        c1_layers = []
        n_min_layers = min(len(p0_layers), len(p1_layers))
        n_max_layers = max(len(p0_layers), len(p1_layers))

        for i in range(n_min_layers):
            # Construct offspring o0 & o1 and incorporate them in the children
            o0, o1 = p0_layers[i].param_crossover(p1_layers[i])
            c0_layers.append(o0)
            c1_layers.append(o1)

        # Append remaining, unmatched layers as-is to the first child's layers
        if len(p0_layers) > len(p1_layers):
            for i in range(n_min_layers, n_max_layers):
                c0_layers.append(p0_layers[i])
        else:
            for i in range(n_min_layers, n_max_layers):
                c0_layers.append(p1_layers[i])

        return (c0_layers, c1_layers)

    def add_random_layer(self, limits):
        """
        Adds a randomly initialised convolution block or dense layer if the
        layer limit allows it.
        """
        if randint(2) == 0:
            if len(self.conv_blocks) < limits.n_conv_blocks_max:
                self.conv_blocks.append(ConvBlockGene.random(limits))
        elif len(self.dense_layers) < limits.n_dense_layers_max:
            self.dense_layers.append(DenseLayerGene.random(limits))

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

    def modify_parameters(self, limits, polynom_mutation_eta=20):
        """
        Modifies all genome parameters with equal probability within an
        interval specific to each parameter.
        """
        n = len(self.conv_blocks)*6 + len(self.dense_layers)*1
        if (len(self.conv_blocks)) == 0:
            n += 3  # To modify self.pool_before_dense parameters
            self.pool_before_dense.mutate(1/n, limits)
        mut_prob = 1/n

        for conv_block in self.conv_blocks:
            conv_block.mutate(mut_prob, limits, eta=polynom_mutation_eta)
        for dense_layer in self.dense_layers:
            dense_layer.mutate(mut_prob, limits, eta=polynom_mutation_eta)

    def phenotype(self, hw=False):
        """
        Constructs and returns the phenotype corresponding to this genome, i.e.
        its expression as a Keras CNN model.
        """
        self.force_valid_cnn()

        inputs = Input(shape=(700, 1))

        for (i, gene) in enumerate(self.conv_blocks):
            # The first layer call has to be on the `inputs` object. Note that there may be 0 conv blocks, resulting in 0 layer calls.
            if i == 0:
                x = Conv1D(gene.n_filters, gene.filter_size, kernel_initializer='he_uniform', activation='selu', padding='same', name=f'block{i}_conv')(inputs)
            else:
                x = Conv1D(gene.n_filters, gene.filter_size, kernel_initializer='he_uniform', activation='selu', padding='same', name=f'block{i}_conv')(x)

            if gene.batch_norm:
                x = BatchNormalization(name=f'block{i}_batchnorm')(x)
            pool_func = AveragePooling1D if gene.pooling.pool_type == PoolType.AVERAGE else MaxPooling1D
            x = pool_func(gene.pooling.pool_size, gene.pooling.pool_stride, name=f'block{i}_pool')(x)

        # Call a pooling layer if there are no convolution blocks
        if len(self.conv_blocks) == 0:
            pool_func = AveragePooling1D if self.pool_before_dense.pool_type == PoolType.AVERAGE else MaxPooling1D
            x = pool_func(self.pool_before_dense.pool_size, self.pool_before_dense.pool_stride, name=f'pre_dense_pool')(inputs)

        x = Flatten()(x)

        for (i, gene) in enumerate(self.dense_layers):
            x = Dense(gene.n_neurons, kernel_initializer=gene.weight_init, activation=gene.act_func, name=f'dense{i}')(x)

        x = Dense(9 if hw else 256, activation='softmax', name='output_layer')(x)

        return Model(inputs, x)

    def force_valid_cnn(self):
        """
        Enforces parameter values of ConvBlock genes so that the resulting CNN
        is valid, i.e. so that it never produces a feature map of size 0.
        """
        self.correct_pool_sizes()

        # Remove layers that still don't have valid parameters
        self.conv_blocks = [
            block for block in self.conv_blocks
            if block.pooling.pool_size != 0
        ]

    def correct_pool_sizes(self):
        """
        Adjusts the pool size of all convolution blocks where this is necessary
        due to the pool size being larger than the previous layer's feature map
        size.
        """
        inp_size = 700
        for conv_block in self.conv_blocks:
            conv_block.pooling.pool_size = min(conv_block.pooling.pool_size, inp_size)
            inp_size = conv_block.output_map_size(inp_size)

    def clone(self):
        """
        Returns a duplicate clone (by value) of this genome.
        """
        clone = CnnGenome()
        clone.conv_blocks = deepcopy(self.conv_blocks)
        clone.dense_layers = deepcopy(self.dense_layers)
        clone.pool_before_dense = self.pool_before_dense.clone()

        return clone

    def prettyprint(self):
        """
        Prints a formatted string of this genome's parameters.
        """
        print(
            f"""
            Genome summary:
            {len(self.conv_blocks)} convolution blocks
            {len(self.dense_layers)} dense layers
            """
        )

        if len(self.conv_blocks) == 0:
            print("="*30)
            self.pool_before_dense.prettyprint()

        for (i, conv_block) in enumerate(self.conv_blocks):
            print("="*30)
            conv_block.prettyprint(i)

        for (i, dense_layer) in enumerate(self.dense_layers):
            print("="*30)
            dense_layer.prettyprint(i)


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

        self.pooling = PoolingGene(pool_type, pool_size, pool_stride)

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
            pool_type=PoolType.random(),
            pool_size=randint(limits.pool_size_min, limits.pool_size_max + 1),
            pool_stride=randint(limits.pool_stride_min, limits.pool_stride_max + 1)
        )

    def mutate(self, mut_prob, limits, eta=20):
        """
        Mutates all of this gene's parameters with probability `mut_prob`
        within limits specific to each parameter.
        """
        self.n_filters = int(apply_polynom_mutation_with_prob(
            self.n_filters, limits.n_filters_min, limits.n_filters_max, eta, mut_prob))
        self.filter_size = int(apply_polynom_mutation_with_prob(
            self.filter_size, limits.filter_size_min, limits.filter_size_max, eta, mut_prob))

        self.batch_norm = apply_boolean_mutation_with_prob(self.batch_norm, mut_prob)

        self.pooling.mutate(mut_prob, limits, eta)

    def param_crossover(self, other):
        """
        Constructs two offspring genes through crossover on this and the
        `other` ConvBlockGene by iterating over each parameter, randomly
        setting one child's parameter value to that of one parent and setting
        the other child's value to that of the other parent.
        """
        g0, g1 = ConvBlockGene(), ConvBlockGene()

        g0.n_filters, g1.n_filters = randomise_order(self.n_filters, other.n_filters)
        g0.filter_size, g1.filter_size = randomise_order(self.filter_size, other.filter_size)
        g0.batch_norm, g1.batch_norm = randomise_order(self.batch_norm, other.batch_norm)
        g0.pooling, g1.pooling = self.pooling.param_crossover(other.pooling)

        return (g0, g1)

    def output_map_size(self, n_inputs):
        """
        Returns the size the first dimension of the feature map would have
        after applying this convolution block to a feature map with `n_inputs`
        inputs. 
        """
        output_size = int(n_inputs/self.pooling.pool_stride)
        if self.pooling.pool_size > self.pooling.pool_stride:
            output_size -= int(self.pooling.pool_size/self.pooling.pool_stride)
        return output_size

    def clone(self):
        """
        Returns a duplicate clone (by value) of this gene.
        """
        return ConvBlockGene(
            self.n_filters, self.filter_size, self.batch_norm,
            self.pooling.pool_type, self.pooling.pool_size,
            self.pooling.pool_stride
        )

    def prettyprint(self, block_nr):
        """
        Prints a formatted string of this gene's parameters.
        """
        bn_str = "BatchNorm" if self.batch_norm else "No BatchNorm"
        print(f"Conv block {block_nr}:\n{self.n_filters} filters\n" + \
              f"filter size {self.filter_size}\n{bn_str}")
        self.pooling.prettyprint()


class PoolingGene:
    """
    A pooling layer gene describing the parameter of a pooling layer used in a
    neural network.
    """
    def __init__(self, pool_type=PoolType.AVERAGE, pool_size=2, pool_stride=2):
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    @staticmethod
    def random(limits):
        return PoolingGene(
            pool_type=PoolType.random(),
            pool_size=randint(limits.pool_size_min, limits.pool_size_max + 1),
            pool_stride=randint(limits.pool_stride_min, limits.pool_stride_max + 1)
        )

    def mutate(self, mut_prob, limits, eta=20):
        """
        Mutates all of this gene's parameters with probability `mut_prob`
        within limits specific to each parameter.
        """
        self.pool_type = self.pool_type.mutate_with_prob(mut_prob)
        self.pool_size = int(apply_polynom_mutation_with_prob(
            self.pool_size, limits.pool_size_min, limits.pool_size_max, eta, mut_prob))
        self.pool_stride = int(apply_polynom_mutation_with_prob(
            self.pool_stride, limits.pool_stride_min, limits.pool_stride_max, eta, mut_prob))

    def param_crossover(self, other):
        """
        Constructs two offspring genes through crossover on this and the
        `other` PoolingGene by iterating over each parameter, randomly setting
        one child's parameter value to that of one parent and setting the other
        child's value to that of the other parent.
        """
        g0, g1 = PoolingGene(), PoolingGene()

        g0.pool_type, g1.pool_type = randomise_order(self.pool_type, other.pool_type)
        g0.pool_size, g1.pool_size = randomise_order(self.pool_size, other.pool_size)
        g0.pool_stride, g1.pool_stride = randomise_order(self.pool_stride, other.pool_stride)

        return (g0, g1)

    def clone(self):
        """
        Returns a duplicate clone (by value) of this gene.
        """
        return PoolingGene(self.pool_type, self.pool_size, self.pool_stride)

    def prettyprint(self):
        """
        Prints a formatted string of this gene's parameters.
        """
        print(f"{self.pool_type.name.capitalize()}Pooling\n" + \
              f"pool size = {self.pool_size}\n" + \
              f"pool stride = {self.pool_stride}")


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

    def mutate(self, mut_prob, limits, eta=20):
        """
        Mutates all of this gene's parameters with probability `mut_prob`
        within limits specific to each parameter.
        """
        self.n_neurons = int(apply_polynom_mutation_with_prob(
            self.n_neurons, limits.n_dense_neurons_min,
            limits.n_dense_neurons_max, eta, mut_prob
        ))

    def param_crossover(self, other):
        """
        Constructs two offspring genes through crossover on this and the
        `other` DenseLayergene by iterating over each parameter, randomly
        setting one child's parameter value to that of one parent and setting
        the other child's value to that of the other parent.
        """
        g0, g1 = DenseLayerGene(), DenseLayerGene()

        g0.n_neurons, g1.n_neurons = randomise_order(self.n_neurons, other.n_neurons)

        return (g0, g1)

    def clone(self):
        """
        Returns a duplicate clone (by value) of this gene.
        """
        return DenseLayerGene(self.n_neurons)

    def prettyprint(self, layer_nr):
        """
        Prints a formatted string of this gene's parameters.
        """
        print(f"Dense layer {layer_nr}:\n{self.n_neurons} neurons")


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
        return apply_polynomial_mutation(x, lo, hi, eta)
    return x


def apply_boolean_mutation_with_prob(b, mut_prob):
    """
    Sets bool `b` to the opposite of its current boolean value with probability
    `mut_prob`.
    """
    return not b if np.random.uniform() < mut_prob else b


def randomise_order(x0, x1):
    """
    Returns a tuple of given variables `x0` and `x1` in random order.
    """
    return (x0, x1) if randint(2) == 0 else (x1, x0)

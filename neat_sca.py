import configparser
import multiprocessing as mp

import neat
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_processing import (load_chipwhisperer_data, load_prepared_ascad_vars,
                             sample_traces)
from helpers import (get_pool_size, neat_nn_predictions, compute_fitness,
                     consecutive_int_groups)
from metrics import MetricType


x, y, pt, k, k_idx, g_hw = None, None, None, None, 1, True
metric = MetricType.CATEGORICAL_CROSS_ENTROPY
num_folds = 1


class NeatSca:
    def __init__(self, pop_size, max_gens, config_filepath="./neat-config",
                 remote=False, parallelise=True):
        self.pop_size = pop_size
        self.max_gens = max_gens

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_filepath)
        self.config.pop_size = pop_size

        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(False))

        self.parallelise = parallelise
        if parallelise:
            pool_size = get_pool_size(remote, pop_size//2)
            print(f"Initialising process pool with pool size = {pool_size}")
            self.pe = neat.ParallelEvaluator(pool_size, evaluate_genome_fitness)

    def run(self, x_train, y_train, pt_train, k_train, hw=True, n_folds=1):
        """
        Runs NEAT according to a predefined config file on the given training
        data.

        Returns:
            A tuple containing the final best genome and the config object.
        """
        # Global trace set can be either static or a list of folds
        global x, y, pt, k, g_hw, num_folds
        x, y, pt, k, g_hw = x_train, y_train, pt_train, k_train, hw
        num_folds = n_folds
        set_global_data("cw", 2000, subkey_idx=1, n_folds=1, remote=False, hw=True, metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY)

        eval_func = self.pe.evaluate if self.parallelise else eval_pop_fitness
        best_indiv = self.population.run(eval_func, self.max_gens)

        return (best_indiv, self.config)

    def get_results(self):
        # TODO: Obtain best fitness per gen and top N indivs
        pass


def evaluate_genome_fitness(genome, config):
    """
    Evaluates and returns the fitness of the given genome. This method assumes
    the globally available traces are a static set.

    Arguments:
        genome: A (genome_id, genome) tuple.
    """
    global x, y, pt, k, k_idx, g_hw, metric

    nn = neat.nn.FeedForwardNetwork.create(genome, config)
    preds = neat_nn_predictions(nn, x, g_hw)

    fit = float(
        compute_fitness(nn, x, y, pt, metric, k, len(x), k_idx, g_hw, preds)
    )
    return -fit


def eval_pop_fitness(genomes, config):
    for (i, genome) in enumerate(genomes):
        genomes[i][1].fitness = evaluate_genome_fitness(genome[1], config)


def multifold_genome_fitness_eval(genome, config):
    """
    Evaluates and returns the fitness of the given genome. This method assumes
    the globally available data set objects are lists containing multiple
    folds of the original data set.

    Arguments:
        genome: A (genome_id, genome) tuple.
    """
    global x, y, pt, k, k_idx, g_hw, metric, num_folds
    assert x is not None

    nn = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = np.zeros(num_folds, dtype=np.float64)
    for i in range(num_folds):
        preds = neat_nn_predictions(nn, x[i], g_hw)
        fitnesses[i] = compute_fitness(
            nn, x[i], y[i], pt[i], metric, k, len(x), k_idx, g_hw, preds)

    return -float(np.mean(fitnesses))


def genome_to_keras_model(genome, config, use_genome_params=False):
    """
    Converts a NEAT-Python genome instance to a keras model using the keras
    functional API.
    """
    layers = neat.graphs.feed_forward_layers(
        config.genome_config.input_keys, config.genome_config.output_keys,
        genome.connections
    )

    n_inputs = config.genome_config.num_inputs
    inputs = keras.Input(shape=(n_inputs, 1))
    # x = keras.layers.Flatten()(inputs)

    # # # Store all inputs & intermediate neuron outputs in a dict, represented as
    # # # individual keras layers.
    # # node_outputs = {
    # #     -(i + 1): keras.layers.Lambda(lambda l: l[:, i], output_shape=(1,))(x)
    # #     for i in range(n_inputs)
    # # }

    node_outputs = {}
    for layer in layers:
        for node_id in layer:
            # Construct connections separately for inputs & hidden nodes
            inc_input_idxs = []
            inc_hidden_node_ids = []
            inc_weights = []  # TODO: Leave weights either unused or optional
            for (conn_id, conn) in genome.connections.items():
                if not conn.enabled:
                    continue

                start_id, end_id = conn_id
                if end_id == node_id:
                    # Differentiate between input nodes and hidden nodes
                    if start_id < 0:
                        true_idx = -start_id - 1
                        inc_input_idxs.append(true_idx)
                    else:
                        inc_hidden_node_ids.append(start_id)

                    if use_genome_params:
                        inc_weights.append(conn.weight)

            # Ignore nodes without incoming connections
            if not inc_input_idxs and not inc_hidden_node_ids:
                continue

            node = genome.nodes[node_id]

            # Construct input layer objects to filter disabled connections
            input_idx_groups = consecutive_int_groups(inc_input_idxs)  # TODO: Determine if this is always sorted
            input_layers = [
                keras.layers.Flatten()(inputs[:, idxs[0]:(idxs[-1] + 1), :])
                for idxs in input_idx_groups
            ]

            # Construct hidden node outputs, which are already layer objects
            hidden_layers = [node_outputs[i] for i in inc_hidden_node_ids]

            # Concatenate all incoming layers
            incoming = None
            if not hidden_layers and len(input_layers) == 1:
                incoming = input_layers[0]
            elif not input_layers and len(hidden_layers) == 1:
                incoming = hidden_layers[0]
            else:
                incoming = keras.layers.Concatenate()(input_layers + hidden_layers)

            kernel_init = keras.initializers.he_uniform()
            bias_init = keras.initializers.zeros()

            if use_genome_params:
                kernel_init = keras.initializers.Constant(inc_weights)
                bias_init = keras.initializers.Constant(node.bias)

            # Define the layer computation for this neuron
            node_outputs[node_id] = keras.layers.Dense(
                1, activation=node.activation,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init
            )(incoming)

    # Concatenate outputs so we can apply softmax on the complete layer
    final_outputs = keras.layers.Concatenate()(
        [node_outputs[i] for i in range(config.genome_config.num_outputs)]  # TODO: Test with generator object if this works. Same for previous list usages.
    )
    final_outputs = keras.layers.Softmax()(final_outputs)

    return keras.Model(inputs, final_outputs)


def set_global_data(dataset_name, n_traces, subkey_idx, n_folds=1,
                    remote=False, hw=True,
                    metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY,
                    balanced=False):
    mapping = {
        "ascad": load_prepared_ascad_vars,
        "cw": load_chipwhisperer_data
    }

    DATA_LOAD_FUNC = mapping[dataset_name]
    data = DATA_LOAD_FUNC(
        subkey_idx=subkey_idx, remote=remote, hw=hw
    )

    global x, y, pt, k, k_idx, g_hw, metric, num_folds
    x, y, pt, k = data[0], data[1], data[2], data[-1]
    k_idx, g_hw, metric, num_folds = subkey_idx, hw, metric_type, n_folds

    n_cls = 9 if hw else 256
    x, y, pt = sample_traces(n_traces, x, y, pt, n_cls, balanced=balanced)

    if metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        y = tf.keras.utils.to_categorical(y, (9 if hw else 256))

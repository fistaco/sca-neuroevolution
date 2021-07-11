import configparser
import multiprocessing as mp
import pickle

import neat
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.python.ops.gen_nn_ops import avg_pool

from data_processing import (load_chipwhisperer_data, load_prepared_ascad_vars,
                             sample_traces, load_data)
from helpers import (get_pool_size, neat_nn_predictions, compute_fitness,
                     consecutive_int_groups, is_categorical)
from metrics import MetricType
from models import constant_zero_tensor


x, y, pt, k, k_idx, g_hw = None, None, None, None, 1, True
metric = MetricType.CATEGORICAL_CROSS_ENTROPY
sgd_train = True
avg_pooling = True
num_folds = 1


class NeatSca:
    def __init__(self, pop_size, max_gens, config_filepath="./neat-config",
                 remote=False, parallelise=True):
        global x, g_hw, avg_pooling

        self.pop_size = pop_size
        self.max_gens = max_gens

        n_inputs = len(x[0]) if not avg_pooling else len(x[0])//2
        n_outputs = 9 if g_hw else 256

        # Modify config file according to the given parameters
        # config = configparser.ConfigParser()
        # config.read("neat-config")
        # config["NEAT"]["pop_size"] = str(pop_size)
        # config["DefaultGenome"]["num_inputs"] = str(n_inputs)
        # config["DefaultGenome"]["num_outputs"] = str(n_outputs)
        # with open(config_filepath, "w") as f:
        #     config.write(f)

        # TODO: Modify config file according to parameters before loading
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_filepath)
        self.config.pop_size = pop_size
        self.config.genome_config.num_inputs = n_inputs
        self.config.genome_config.num_outputs = n_outputs

        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(False))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)

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
        # global x, y, pt, k, g_hw, num_folds, sgd_train, avg_pooling
        # x, y, pt, k, g_hw = x_train, y_train, pt_train, k_train, hw
        # num_folds = n_folds
        # set_global_data("cw", 8000, subkey_idx=1, n_folds=1, remote=False, hw=True, metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY)

        eval_func = self.pe.evaluate if self.parallelise else eval_pop_fitness
        best_indiv = self.population.run(eval_func, self.max_gens)

        if self.parallelise:
            del self.pe

        return (best_indiv, self.config)

    def get_results(self):
        """
        Returns a tuple containing the best fitness per generation and the top
        ten individuals from the most recent NEAT run.
        """
        top_ten = self.stats.best_unique_genomes(10)
        best_fitness_per_gen = [g.fitness for g in self.stats.most_fit_genomes]

        return (best_fitness_per_gen, top_ten)


def evaluate_genome_fitness(genome, config):
    """
    Evaluates and returns the fitness of the given genome. This method assumes
    the globally available traces are a static set.

    Arguments:
        genome: A (genome_id, genome) tuple.
    """
    global x, y, pt, k, k_idx, g_hw, metric, sgd_train, avg_pooling

    # nn = neat.nn.FeedForwardNetwork.create(genome, config)
    # preds = neat_nn_predictions(nn, x, g_hw)
    nn = genome_to_keras_model(genome, config, use_avg_pooling=avg_pooling)

    if nn is None:
        return -777  # Impossibly low fitness regardless of metric

    if sgd_train:
        y_cat = y
        if not is_categorical(y):
            y_cat = keras.utils.to_categorical(y)
        optimizer = keras.optimizers.Adam(learning_rate=5e-3)
        loss_fn = keras.losses.CategoricalCrossentropy()
        nn.compile(optimizer, loss_fn)
        history = nn.fit(x, y_cat, batch_size=50, epochs=50, verbose=0)

    fit = float(
        compute_fitness(nn, x, y, pt, metric, k, len(x), k_idx, g_hw)
    )
    return -fit


def eval_pop_fitness(genomes, config):
    for (i, genome) in enumerate(genomes):
        with open("most_recent_genome.pickle", "wb") as f:
            pickle.dump(genome, f)
        genomes[i][1].fitness = evaluate_genome_fitness(genome[1], config)


def multifold_genome_fitness_eval(genome, config):
    """
    Evaluates and returns the fitness of the given genome. This method assumes
    the globally available data set objects are lists containing multiple
    folds of the original data set.

    Arguments:
        genome: A (genome_id, genome) tuple.
    """
    global x, y, pt, k, k_idx, g_hw, metric, num_folds, sgd_train, avg_pooling
    assert x is not None

    nn = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = np.zeros(num_folds, dtype=np.float64)
    for i in range(num_folds):
        preds = neat_nn_predictions(nn, x[i], g_hw)
        fitnesses[i] = compute_fitness(
            nn, x[i], y[i], pt[i], metric, k, len(x), k_idx, g_hw, preds)

    return -float(np.mean(fitnesses))


def genome_to_keras_model(genome, config, use_genome_params=False,
                          use_avg_pooling=True):
    """
    Converts a NEAT-Python genome instance to a keras model using the keras
    functional API.
    """
    layers = neat.graphs.feed_forward_layers(
        config.genome_config.input_keys, config.genome_config.output_keys,
        genome.connections
    )

    if not layers:
        return None

    n_inputs = config.genome_config.num_inputs if not use_avg_pooling \
        else config.genome_config.num_inputs*2
    inputs = keras.Input(shape=(n_inputs, 1))

    init_layer = inputs
    if use_avg_pooling:
        init_layer = keras.layers.AveragePooling1D(
            pool_size=2, strides=2, input_shape=(n_inputs, 1)
        )(inputs)

    unreachable = {}  # Track unreachable nodes to terminate interrupted paths

    node_outputs = {}
    for layer in layers:
        for node_id in layer:
            # Construct connections separately for inputs & hidden nodes
            inc_input_idxs = []
            inc_hidden_node_ids = []
            inc_weights = []
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
                        if not start_id in unreachable:
                            inc_hidden_node_ids.append(start_id)

                    if use_genome_params:
                        inc_weights.append(conn.weight)

            # Ignore nodes without incoming connections
            if not inc_input_idxs and not inc_hidden_node_ids:
                unreachable[271] = True
                continue

            node = genome.nodes[node_id]

            # Construct input layer objects to filter disabled connections
            inc_input_idxs = np.sort(inc_input_idxs)
            input_idx_groups = consecutive_int_groups(inc_input_idxs)
            input_layers = [
                Flatten()(init_layer[:, idxs[0]:(idxs[-1] + 1), :])
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
                incoming = Concatenate()(input_layers + hidden_layers)

            kernel_init = keras.initializers.glorot_uniform()
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
    outputs = []
    for i in range(config.genome_config.num_outputs):
        if i in node_outputs:
            outputs.append(node_outputs[i])
        else:
            # Use a dummy output of 0 to keep categorical label outputs intact
            outputs.append(keras.layers.Lambda(constant_zero_tensor)(inputs))
    final_output_layer = keras.layers.Concatenate()(outputs)
    final_output_layer = keras.layers.Softmax()(final_output_layer)

    return keras.Model(inputs, final_output_layer)


def set_global_data(dataset_name, n_traces, subkey_idx, n_folds=1,
                    remote=False, hw=True,
                    metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY,
                    balanced=False, use_sgd=True, use_avg_pooling=True):
    data = load_data(dataset_name, hw, remote)

    global x, y, pt, k, k_idx, g_hw, metric, num_folds, sgd_train, avg_pooling
    x, y, pt, k = data[0], data[1], data[2], data[3]
    k_idx, g_hw, metric, num_folds = subkey_idx, hw, metric_type, n_folds
    sgd_train, avg_pooling = use_sgd, use_avg_pooling

    n_cls = 9 if hw else 256
    x, y, pt = sample_traces(n_traces, x, y, pt, n_cls, balanced=balanced)

    if metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        y = tf.keras.utils.to_categorical(y, (9 if hw else 256))

import configparser
import multiprocessing as mp
from multiprocessing import pool
import pickle

import graphviz
import neat
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dense, Flatten, LeakyReLU
from tensorflow.python.ops.gen_array_ops import reverse

from data_processing import (sample_traces, load_data, to_hw)
from helpers import (get_pool_size, neat_nn_predictions, compute_fitness,
                     consecutive_int_groups, is_categorical)
from metrics import MetricType
from models import constant_zero_tensor


x, y, pt, k, k_idx, g_hw = None, None, None, None, 1, True
metric = MetricType.CATEGORICAL_CROSS_ENTROPY
sgd_train = True
avg_pooling = True
pool_size = 2
num_folds = 1
x_val, y_val, pt_val = None, None, None


class NeatSca:
    def __init__(self, pop_size, max_gens, config_filepath="./neat-config",
                 remote=False, parallelise=True, only_evolve_hidden=False,
                 fs_neat=False, comp_thresh=None, tselect=False,
                 double_layer_init=False):
        global x, g_hw, avg_pooling, pool_size

        self.pop_size = pop_size
        self.max_gens = max_gens

        n_inputs = len(x[0]) if not avg_pooling else len(x[0])//pool_size
        n_outputs = 9 if g_hw else 256

        self.config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_filepath,
            only_evolve_hidden
        )
        self.config.pop_size = pop_size
        self.config.genome_config.num_inputs = n_inputs
        self.config.genome_config.input_keys = [-i-1 for i in range(n_inputs)]
        self.config.genome_config.num_outputs = n_outputs
        self.config.genome_config.output_keys = list(range(n_outputs))
        self.config.genome_config.initial_connection = "full_nodirect" \
            if not fs_neat else "fs_neat_hidden"

        if comp_thresh is None:
            n_hidden = self.config.genome_config.num_hidden
            n_nodes = n_hidden + n_outputs
            n_conns = n_inputs*n_hidden + n_hidden*n_outputs
            if fs_neat:
                n_conns = n_hidden + n_outputs

            # Max. 2 nodes and 3 conns are added to a genome each generation
            comp_thresh = 5*(2/n_nodes + 3/n_conns)
        # comp_thresh = 0.53 if g_hw else 0.04
        self.config.species_set_config.compatibility_threshold = comp_thresh

        self.population = neat.Population(self.config, tourn_select=tselect)
        # with open("most_recent_neat_genome.pickle", "wb") as f:
        #     pickle.dump(self.population.population[1], f)
        # exit()
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
    global x, y, pt, k, k_idx, g_hw, metric, sgd_train, avg_pooling, pool_size
    global x_val, y_val, pt_val

    # Set seeds to ensure equal opportunity in SGD training
    tf.random.set_seed(77)
    np.random.seed(77)

    # nn = neat.nn.FeedForwardNetwork.create(genome, config)
    # preds = neat_nn_predictions(nn, x, g_hw)
    nn = genome_to_keras_model(
        genome, config, use_avg_pooling=avg_pooling, pool_param=pool_size
    )

    if nn is None:
        return -777  # Impossibly low fitness regardless of metric

    if sgd_train:
        y_cat = y
        if not is_categorical(y):
            y_cat = keras.utils.to_categorical(y)
        optimizer = keras.optimizers.Adam(learning_rate=5e-3)
        loss_fn = keras.losses.CategoricalCrossentropy()
        nn.compile(optimizer, loss_fn)
        history = nn.fit(x, y_cat, batch_size=100, epochs=30, verbose=0)

    if x_val is None:
        fit = float(compute_fitness(
            nn, x, y, pt, metric, k, len(x), k_idx, g_hw
        ))
    else:
        fit = float(compute_fitness(
            nn, x_val, y_val, pt_val, metric, k, len(x), k_idx, g_hw
        ))
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
                          use_avg_pooling=True, pool_param=2):
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
        else config.genome_config.num_inputs*pool_param
    inputs = keras.Input(shape=(n_inputs, 1))

    init_layer = inputs
    if use_avg_pooling:
        init_layer = keras.layers.AveragePooling1D(
            pool_size=pool_param, strides=pool_param, input_shape=(n_inputs, 1)
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
                unreachable[node_id] = True
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


def construct_neat_config(input_size, hw, config_filepath, only_evolve_hidden,
                          pool_size=1):
    """
    Constructs a `Config` object compatible with the NEAT-Python module for
    usage with the `genome_to_keras_model` function.
    """
    n_inputs = input_size//pool_size
    n_outputs = 9 if hw else 256

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_filepath,
        only_evolve_hidden
    )
    config.genome_config.num_inputs = n_inputs
    config.genome_config.input_keys = [-i-1 for i in range(n_inputs)]
    config.genome_config.num_outputs = n_outputs
    config.genome_config.output_keys = list(range(n_outputs))

    return config


def draw_common_genome_nn_structures(genomes, label=""):
    """
    Draws the common elements of NNs correspondging to a given list of genomes
    by finding in- and output nodes that are connected to non-initial hidden
    nodes.
    """
    pass


def draw_simple_genome_nn(genome, label=""):
    """
    Draws the NN corresponding to the given `genome` by drawing all nodes and
    connections except the ones that were present in the unevolved genome.
    """
    dot = graphviz.Digraph("NEAT NN visualisation", format="png")


def draw_genome_nn(genome, label="", only_draw_hidden=True, n_outputs=256,
                   draw_inp_to_hid=True, draw_new_hid_to_out=True):
    """
    Draws the NN corresponding to the given `genome` as a graph.
    """
    dot = graphviz.Digraph("NEAT NN visualisation", format="png")

    nodes = genome.nodes.keys()
    edges = genome.connections.items()

    if only_draw_hidden:
        # Remove output nodes. Input nodes are not present in the first place.
        nodes = {n for n in nodes if n >= n_outputs}

        # Force the tail of each edge to be a hidden node
        edges = set([
            ((i, o), c) for ((i, o), c) in edges
            if i >= n_outputs and o >= n_outputs and c.enabled
        ])

        # Map each node to its connections
        node_inc_conns = {n:[] for n in nodes}
        node_out_conns = {n:[] for n in nodes}
        for ((i, o), c) in genome.connections.items():
            if not c.enabled:
                continue
            if o in node_inc_conns:
                node_inc_conns[o].append(((i, o), c))
            if i in node_out_conns:
                node_out_conns[i].append(i)

        # Remove nodes with only 1 incoming and outgoing connections
        pass # TODO
            
        # Remove nodes and edges with only 1 incoming and outgoing connection

        if draw_inp_to_hid:
            init_hidden_nodes = []
            inc_conns_per_node = {n:[] for n in nodes}

            # Determine incoming inputs for each hidden node
            for ((i, o), c) in genome.connections.items():
                if not c.enabled:
                    continue
                if o in inc_conns_per_node:
                    inc_conns_per_node[o].append(((i, o), c))

            # Draw nodes/conns with a reasonable number of incoming conns
            for n in inc_conns_per_node:
                inc_conns = inc_conns_per_node[n]

                if len(inc_conns) < 200:
                    edges.update(inc_conns)
                    for ((i, _), _) in inc_conns:
                        if i < 0:
                            nodes.add(i)  # Draw required input layer node
                else:
                    # Connect the "inputs" node to initial FC hid. nodes later
                    init_hidden_nodes.append(n)
        
            if draw_new_hid_to_out:
                # Draw connections from new hidden nodes to output nodes
                for ((i, o), c) in genome.connections.items():
                    if not c.enabled:
                        continue
                    if o < n_outputs and i >=0 and i not in init_hidden_nodes:
                        edges.add(((i, o), c))

        # Divide nodes in ranks for readability
        node_subgraph(dot, nodes, hi=0, order=True, reverse_order=True)
        node_subgraph(dot, init_hidden_nodes)
        hid_layer_start = n_outputs + len(init_hidden_nodes)
        node_subgraph(dot, nodes, lo=hid_layer_start, same_rank=False)
        node_subgraph(dot, nodes, lo=0, hi=n_outputs, order=True)

    for ((i, j), _) in edges:
        dot.edge(str(i), str(j))

    dot.render(f"fig/neat-nn-vis-{label}", view=False)


def node_subgraph(dot, nodes, lo=-np.inf, hi=np.inf, same_rank=True,
                  order=False, reverse_order=False):
    """
    Constructs a subgraph in the given `dot` visualisation by adding IDs from
    the given `nodes` list if they fall within `lo` <= n < `hi`.
    """
    with dot.subgraph() as s:
        if same_rank:
            s.attr(rank="same")

        node_ids = []
        for node_id in nodes:
            if lo <= node_id < hi:
                s.node(str(node_id), str(node_id))
                node_ids.append(node_id)

        if order:
            s.attr("edge", style="invis")
            s.attr(rankdir="LR")

            node_ids = sorted(node_ids, reverse=reverse_order)
            for i in range(len(node_ids) - 1):
                s.edge(str(node_ids[i]), str(node_ids[i + 1]))


def evolve_binary_nn(output_class, hw=False):
    """
    Evolves a binary NN for the given output class by running NEAT with
    training data containing a 50-50 split of traces with and without the
    given output class as a label.
    """
    # TODO:
    # - Implement data splitting in set_global_data -> 3584 traces from one class and 3584 balanced traces from other classes
    # - Ensure a good log-loss model can be trained by creating a custom keras model with 2x10 hidden nodes or smth
    # - Set n_output_nodes to 1 with a custom argument
    # - Determine n_gens and psize for the resources we have
    pass


def train_stacked_model(model_dir):
    """
    Stacks the models in the given `model_dir` by obtaining those models'
    outputs for the given training data and using those as inputs for a
    logistic regression model to produce a final label output.
    """
    pass


def set_global_data(dataset_name, n_traces, subkey_idx, n_folds=1,
                    remote=False, hw=True,
                    metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY,
                    balanced=False, use_sgd=True, use_avg_pooling=True,
                    pool_param=2, seed=None, balance_on_hw=False,
                    noise=0.0, desync=0, n_valid_set=0):
    if seed:
        np.random.seed(seed)

    # Always load ID labels, unless not balancing or balancing on HW
    load_hw_labels = hw if not balanced or balance_on_hw else False
    data = load_data(dataset_name, load_hw_labels, remote, noise_std=noise,
                     desync=desync)

    global x, y, pt, k, k_idx, g_hw, metric, num_folds, sgd_train, avg_pooling
    global x_val, y_val, pt_val, pool_size
    x, y, pt, k = data[0], data[1], data[2], data[3]
    k_idx, g_hw, metric, num_folds = subkey_idx, hw, metric_type, n_folds
    sgd_train, avg_pooling, pool_size = use_sgd, use_avg_pooling, pool_param

    n_cls = 9 if load_hw_labels else 256

    if n_valid_set > 0:  # Sample a validation set
        x, y, pt, x_val, y_val, pt_val = sample_traces(
            n_traces, x, y, pt, n_cls, balanced=balanced, return_remainder=True
        )
        x_val, y_val, pt_val = sample_traces(
            n_valid_set, x_val, y_val, pt_val, balanced=balanced
        )
    else:
        x, y, pt = sample_traces(n_traces, x, y, pt, n_cls, balanced=balanced)

    if hw and not load_hw_labels:
        y = to_hw(y)

    if metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        y = tf.keras.utils.to_categorical(y, (9 if hw else 256))

        if n_valid_set > 0:
            y_val = tf.keras.utils.to_categorical(y_val, (9 if hw else 256))

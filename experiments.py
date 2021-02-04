import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time

from data_processing import load_ascad_data, load_ascad_atk_variables, \
    sample_data, shuffle_data, scale_inputs
from genetic_algorithm import GeneticAlgorithm
from helpers import exec_sca
from metrics import keyrank
from models import build_small_cnn_ascad, load_small_cnn_ascad
from plotting import plot_gens_vs_fitness, plot_n_traces_vs_key_rank


def ga_grid_search():
    pass


def run_ga(max_gens, pop_size, mut_power, mut_rate, crossover_rate,
           mut_power_decay_rate, truncation_proportion, atk_set_size, nn,
           x_validation, y_validation, ptexts_validation, x_test, y_test,
           ptexts_test, true_subkey, parallelise):
    ga = GeneticAlgorithm(
        max_gens,
        pop_size,
        mut_power,
        mut_rate,
        crossover_rate,
        mut_power_decay_rate,
        truncation_proportion,
        atk_set_size,
        parallelise
    )

    # Obtain the best network resulting from the GA
    start = time()
    best_indiv = \
        ga.run(nn, x_validation, y_validation, ptexts_validation, true_subkey)
    end = time()
    t = int(end-start)
    print(f"Time elapsed: {t}")
    # TODO: Create new model from best individual's weights here and test it
    # best_nn = best_indiv.model

    # # Evaluate the best network's performance on the test set
    # key_rank = exec_sca(best_nn, x_test, y_test, ptexts_test, true_subkey)
    # # TODO: plot generational fitness improvement

    print(f"Key rank on validation set: {best_indiv.fitness}")
    # print(f"Key rank on test set: {key_rank}")


def small_cnn_sgd_sca(save=True, subkey_idx=2):
    # Load the ASCAD data set with 700 points per trace
    PATH = "./../ASCAD_data/ASCAD_databases/ASCAD.h5"
    (x, y, x_atk, y_atk, train_meta, atk_meta) = \
        load_ascad_data(PATH, load_metadata=True)
    original_input_shape = (700, 1)
    x_train, y_train = x[:45000], y[:45000]
    x_valid, y_valid = x[:-5000], y[:-5000]

    # Define hyperparameters
    n_epochs = 50
    batch_size = 50
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # Declare easily accessible variables for relevant metadata
    full_key = atk_meta['key'][0]
    target_subkey = full_key[subkey_idx]
    atk_ptexts = atk_meta['plaintext']

    # Convert labels to one-hot encoding probabilities
    y_train_converted = keras.utils.to_categorical(y_train, num_classes=256)

    # Scale the inputs
    x_train = scale_inputs(x_train)
    x_atk = scale_inputs(x_atk)

    # Reshape the trace input to come in singleton arrays for CNN compatibility
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_atk_reshaped = x_atk.reshape((x_atk.shape[0], x_atk.shape[1], 1))

    # Train CNN
    cnn = build_small_cnn_ascad()
    cnn.compile(optimizer, loss_fn)
    history = cnn.fit(x_train, y_train_converted, batch_size, n_epochs)

    # Save the model if desired
    if save:
        cnn.save('./trained_models/efficient_cnn_ascad_model_17kw.h5')

    # Attack with the trained model
    key_rank = exec_sca(cnn, x_atk_reshaped, y_atk, atk_ptexts, target_subkey, subkey_idx=2)

    print(f"Key rank obtained with efficient CNN on ASCAD: {key_rank}")


def attack_ascad_with_cnn(subkey_idx=2, atk_set_size=10000):
    # Load attack set of 10k ASCAD traces and relevant metadata
    (x_atk, y_atk, target_subkey, atk_ptexts) = \
        load_ascad_atk_variables(for_cnns=True, subkey_idx=2, scale=True)
    x_atk, y_atk = x_atk[:atk_set_size], y_atk[:atk_set_size]

    # Load CNN and attack the traces with it
    # cnn = load_small_cnn_ascad()
    cnn = keras.models.load_model("./trained_models/efficient_cnn_ascad_model_from_github.h5")
    key_rank = exec_sca(cnn, x_atk, y_atk, atk_ptexts, target_subkey, subkey_idx)

    print(f"Key rank = {key_rank}")
    

def tenfold_ascad_atk_with_varying_size(nn, subkey_idx=2, experiment_name=""):
     # Load attack set of 10k ASCAD traces and relevant metadata
    (x_atk, y_atk, target_subkey, atk_ptexts) = \
        load_ascad_atk_variables(for_cnns=True, subkey_idx=2, scale=True)

    # Predict outputs for the full set
    y_pred_probs = nn.predict(x_atk)

    # For each fold, store the key rank for various attack set sizes
    atk_set_sizes = range(1, len(y_atk), 50)
    fold_key_ranks = np.zeros((10, len(atk_set_sizes)), dtype=np.uint8)

    # Reuse subsets of the predictions to simulate attacks over different folds
    for fold in range(10):
        y_pred_probs, atk_ptexts = shuffle_data(y_pred_probs, atk_ptexts)

        # Track the attack set size for which we're currently logging results
        atk_set_size_idx = 0

        # Track the summed log probability of each subkey candidate
        subkey_logprobs = np.zeros(256)

        # Iterate over each list of 256 probabilities in label_pred_probs
        for (i, pred_probs) in enumerate(label_pred_probs):
            if i == atk_set_sizes[atk_set_size_idx]:
                fold_key_ranks[fold, atk_set_size_idx] = keyrank(subkey_logprobs, target_subkey)
                atk_set_size_idx += 1

            pt = atk_ptexts[i][subkey_idx]

            # Convert each label to a subkey and add its logprob to the sum
            for (label, label_pred_prob) in enumerate(pred_probs):
                subkey = label_to_subkey(pt, label)

                # Avoid computing np.log(0), which returns -inf
                logprob = np.log(label_pred_prob) if label_pred_prob > 0 else 0
                subkey_logprobs[subkey] += logprob
    
    # Build a dictionary that contains the mean key rank for each trace amount
    mean_key_ranks = {}
    for (i, atk_set_size) in enumerate(atk_set_sizes):
        mean_rank = np.mean([fold_key_ranks[fold][i] for fold in range(10)])
        mean_key_ranks[atk_set_size] = mean_rank
    
    for (n_traces, rank) in mean_key_ranks:
        print(f"Mean key rank with {n_traces} attack traces: {rank}")

    if experiment_name:
        plot_n_traces_vs_key_rank(mean_key_ranks, experiment_name)

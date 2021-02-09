import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from time import time

from data_processing import load_ascad_data, load_ascad_atk_variables, \
    sample_data, shuffle_data, scale_inputs
from genetic_algorithm import GeneticAlgorithm
from helpers import exec_sca, label_to_subkey, compute_mem_req
from metrics import keyrank
from models import build_small_cnn_ascad, load_small_cnn_ascad
from plotting import plot_gens_vs_fitness, plot_n_traces_vs_key_rank


def ga_grid_search():
    pop_sizes = np.arange(25, 251, 75)  # 4 values
    mut_pows = np.arange(0.01, 0.1, 0.02)  # 5 values
    mut_rates = np.arange(0.01, 0.11, 0.03)  # 4 values
    atk_set_sizes = np.array([2, 16, 128, 1024])  # 4 values
    selection_methods = np.array(["roulette_wheel", "tournament"])  # 2 values
    # TODO: test different weight init versions

    # Mutation power decay rate is assumed to be near optimal
    # Total runs without selection method variation = 320
    for pop_size in pop_sizes:
        for mut_pow in mut_pows:
            for mut_rate in mut_rates:
                for atk_set_size in atk_set_sizes:
                    pass


def run_ga(max_gens, pop_size, mut_power, mut_rate, crossover_rate,
           mut_power_decay_rate, truncation_proportion, atk_set_size, nn,
           x_valid, y_valid, ptexts_valid, x_test, y_test, ptexts_test,
           true_validation_subkey, true_atk_subkey, parallelise,
           experiment_name="test"):
    """
    Runs a genetic algorithm with the given parameters and tests the resulting
    best individual on the given test set. The best individual, best fitnesses
    per generation, and results from the final test are saved to pickle files.
    """
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
        ga.run(nn, x_valid, y_valid, ptexts_valid, true_validation_subkey)
    end = time()
    t = int(end-start)
    print(f"Time elapsed: {t}")
    # TODO: Create new model from best individual's weights here and test it
    # best_nn = best_indiv.model

    ga.save_results(best_indiv, experiment_name)

    # Create a new model from the best individual's weights and properly
    # evaluate it on the test set
    cnn = load_small_cnn_ascad()
    cnn.set_weights(best_indiv.weights)
    tenfold_ascad_atk_with_varying_size(cnn, 2, experiment_name)


    # # Evaluate the best network's performance on the test set
    # key_rank = exec_sca(best_nn, x_test, y_test, ptexts_test, true_subkey)
    # # TODO: plot generational fitness improvement

    print(f"Key rank on validation set: {best_indiv.fitness}")
    # print(f"Key rank on test set: {key_rank}")


def single_ga_experiment():
    (x_train, y_train, x_atk, y_atk, train_meta, atk_meta) = \
        load_ascad_data(load_metadata=True)
    original_input_shape = (700, 1)
    x_train, y_train = x_train[:45000], y_train[:45000]

    # Declare easily accessible variables for relevant metadata
    subkey_idx = 2
    target_train_subkey = train_meta['key'][0][subkey_idx]
    train_ptexts = train_meta['plaintext']
    target_atk_subkey = atk_meta['key'][0][subkey_idx]
    atk_ptexts = atk_meta['plaintext']

    # Convert labels to one-hot encoding probabilities
    y_train_converted = keras.utils.to_categorical(y_train, num_classes=256)

    # Reshape the trace input to come in singleton arrays for CNN compatibility
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_atk = x_atk.reshape((x_atk.shape[0], x_atk.shape[1], 1))

    # Train the CNN by running it through the GA
    cnn = load_small_cnn_ascad()
    run_ga(
        max_gens=5,
        pop_size=2,
        mut_power=0.03,
        mut_rate=0.04,
        crossover_rate=0.5,
        mut_power_decay_rate=0.99,
        truncation_proportion=0.4,
        atk_set_size=1024,
        nn=cnn,
        x_valid=x_train,
        y_valid=y_train,
        ptexts_valid=train_ptexts,
        x_test=x_atk,
        y_test=y_atk,
        ptexts_test=atk_ptexts,
        true_validation_subkey=target_train_subkey,
        true_atk_subkey=target_atk_subkey,
        parallelise=True
    )


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
    # key_rank = exec_sca(cnn, x_atk, y_atk, atk_ptexts, target_subkey, subkey_idx)
    # print(f"Key rank = {key_rank}")

    tenfold_ascad_atk_with_varying_size(cnn)
    

def tenfold_ascad_atk_with_varying_size(nn, subkey_idx=2, experiment_name="",
    atk_data=None):
    # Use the given data if possible. Load 10k ASCAD attack traces otherwise.
    (x_atk, y_atk, target_subkey, atk_ptexts) = \
        atk_data if atk_data \
        else load_ascad_atk_variables(for_cnns=True, subkey_idx=2, scale=True)

    # Predict outputs for the full set
    y_pred_probs = nn.predict(x_atk)

    # For each fold, store the key rank for various attack set sizes
    atk_set_sizes = range(1, len(y_atk), 50)
    n_atk_set_sizes = len(atk_set_sizes)
    fold_key_ranks = np.zeros((10, n_atk_set_sizes), dtype=np.uint8)

    # Reuse subsets of the predictions to simulate attacks over different folds
    for fold in range(10):
        print(f"Obtaining key ranks for fold {fold}...")
        y_pred_probs, atk_ptexts = shuffle_data(y_pred_probs, atk_ptexts)

        # Track the attack set size for which we're currently logging results
        atk_set_size_idx = 0

        # Track the summed log probability of each subkey candidate
        subkey_logprobs = np.zeros(256)

        # Iterate over each list of 256 probabilities in y_pred_probs
        for (i, pred_probs) in enumerate(y_pred_probs):
            if i < n_atk_set_sizes and i == atk_set_sizes[atk_set_size_idx]:
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
        mean_key_ranks[atk_set_size] = int(mean_rank)
    with open(f"{experiment_name}_test_set_mean_key_ranks", "wb") as f:
        pickle.dump(mean_key_ranks, f)
    
    for (n_traces, rank) in mean_key_ranks.items():
        print(f"Mean key rank with {n_traces} attack traces: {rank}")

    if experiment_name:
        plot_n_traces_vs_key_rank(mean_key_ranks, experiment_name)


def compute_memory_requirements(pop_sizes, atk_set_sizes):
    """
    Approximates the required memory (GB) to run the GA for a given list of
    population sizes and attack set sizes.
    """
    cnn = load_small_cnn_ascad()

    for pop_size in pop_sizes:
        for atk_set_size in atk_set_sizes:
            mem = compute_mem_req(pop_size, cnn, atk_set_size)/1e6
            print(f"Required memory for pop size {pop_size} & attack set size {atk_set_size}: {mem} GB")

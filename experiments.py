import os
import pickle
from time import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_processing import (load_ascad_atk_variables, load_ascad_data,
                             sample_data, scale_inputs, shuffle_data,
                             load_prepared_ascad_vars)
from genetic_algorithm import GeneticAlgorithm
from helpers import (compute_mem_req, compute_mem_req_from_known_vals,
                     exec_sca, gen_experiment_name, label_to_subkey,
                     kfold_mean_key_ranks)
from metrics import MetricType, keyrank
from models import (build_small_cnn_ascad, load_nn_from_experiment_results,
                    load_small_cnn_ascad, load_small_cnn_ascad_no_batch_norm,
                    load_small_mlp_ascad, build_small_cnn_ascad_trainable_conv)
from plotting import plot_gens_vs_fitness, plot_n_traces_vs_key_rank


def ga_grid_search():
    pop_sizes = np.arange(25, 251, 75)  # 4 values
    mut_pows = np.arange(0.01, 0.08, 0.02)  # 4 values
    mut_rates = np.arange(0.01, 0.11, 0.03)  # 4 values
    mut_pow_dec_rates = np.array([0.99, 0.999, 1.0]) # 3 values
    fi_dec_rates = np.arange(0.0, 0.5, 0.2) # 3 values
    atk_set_sizes = np.array([8, 16, 64, 256])  # 4 values
    selection_methods = np.array(["roulette_wheel", "tournament"])  # 2 values
    # TODO: test different weight init versions
    # TODO: Only test small atk set sizes with fitness inheritance enabled

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
           true_validation_subkey, true_atk_subkey, subkey_idx, parallelise,
           apply_fi, select_fun, metric_type, experiment_name="test",
           evaluate_on_test_set=True):
    """
    Runs a genetic algorithm with the given parameters and tests the resulting
    best individual on the given test set. The best individual, best fitnesses
    per generation, and results from the final test are saved to pickle files.

    Returns:
        The result tuple of the GA, which has already been saved to a file.
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
        parallelise,
        apply_fi,
        select_fun,
        metric_type
    )

    # Obtain the best network resulting from the GA
    start = time()
    best_indiv = \
        ga.run(nn, x_valid, y_valid, ptexts_valid, true_validation_subkey, subkey_idx)
    end = time()
    t = int(end-start)
    print(f"Time elapsed: {t}")
    # TODO: Create new model from best individual's weights here and test it
    # best_nn = best_indiv.model

    # Save and plot results
    ga_results = ga.save_results(best_indiv, experiment_name)
    plot_gens_vs_fitness(experiment_name, ga.best_fitness_per_gen)

    if evaluate_on_test_set:
        # Create a new model from the best individual's weights and test it
        nn.set_weights(best_indiv.weights)
        kfold_ascad_atk_with_varying_size(
            30,
            nn,
            2,
            experiment_name,
            atk_data=(x_test, y_test, true_atk_subkey, ptexts_test)
        )

        # Evaluate the best network's performance on the test set
        key_rank = exec_sca(nn, x_test, y_test, ptexts_test, true_atk_subkey)
        print(f"Key rank on test set with exec_sca: {key_rank}")

    print(f"Key rank on validation set: {best_indiv.fitness}")
    
    return ga_results


def single_ga_experiment(remote_loc=False, use_mlp=False):
    (x_train, y_train, x_atk, y_atk, train_meta, atk_meta) = \
        load_ascad_data(load_metadata=True, remote_loc=remote_loc)
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

    # Scale all trace inputs to [low, 1]
    low_bound = -1 if use_mlp else 0
    x_train = scale_inputs(x_train, low_bound)
    x_atk = scale_inputs(x_atk, low_bound)

    # Reshape the trace input to come in singleton arrays for CNN compatibility
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_atk = x_atk.reshape((x_atk.shape[0], x_atk.shape[1], 1))

    # Train the CNN by running it through the GA
    nn = load_small_cnn_ascad_no_batch_norm()
    # nn = load_small_mlp_ascad()

    pop_size = 50
    atk_set_size = 16
    select_fun = "tournament"
    run_ga(
        max_gens=100,
        pop_size=pop_size,
        mut_power=0.03,
        mut_rate=0.04,
        crossover_rate=0.5,
        mut_power_decay_rate=0.99,
        truncation_proportion=0.4,
        atk_set_size=atk_set_size,
        nn=nn,
        x_valid=x_train,
        y_valid=y_train,
        ptexts_valid=train_ptexts,
        x_test=x_atk,
        y_test=y_atk,
        ptexts_test=atk_ptexts,
        true_validation_subkey=target_train_subkey,
        true_atk_subkey=target_atk_subkey,
        subkey_idx=subkey_idx,
        parallelise=True,
        apply_fi=True,
        select_fun=select_fun,
        metric_type=MetricType.KEYRANK,
        experiment_name=gen_experiment_name(pop_size, atk_set_size, select_fun)
    )


def averaged_ga_experiment(max_gens, pop_size, mut_power, mut_rate,
           crossover_rate, mut_power_decay_rate, truncation_proportion,
           atk_set_size, nn, x_valid, y_valid, ptexts_valid, x_test, y_test,
           ptexts_test, true_validation_subkey, true_atk_subkey, parallelise,
           apply_fi, select_fun, metric_type, n_experiments=10,
           experiment_name="test"):
    """
    Runs a given amount of GA experiments with the given parameters and returns
    the average key rank obtained with the full given attack set.

    The results of each run are stored in a directory specific to this
    experiment.
    """
    # Create results directory if necessary
    dir_path = f"results/{experiment_name}/"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    avg_keyrank = 0
    for i in range(n_experiments):
        ga = GeneticAlgorithm(
            max_gens,
            pop_size,
            mut_power,
            mut_rate,
            crossover_rate,
            mut_power_decay_rate,
            truncation_proportion,
            atk_set_size,
            parallelise,
            apply_fi,
            select_fun,
            metric_type
        )

        best_indiv = \
            ga.run(nn, x_valid, y_valid, ptexts_valid, true_validation_subkey)

        # Save and plot results # TODO: save results in a directory allocated to this experiment
        ga.save_results(best_indiv, f"{experiment_name}/run-{i}")
        # plot_gens_vs_fitness(ga.best_fitness_per_gen, experiment_name)  # TODO: plot each experiment in the same graph

        # Create a new model from the best individual's weights and evaluate it
        cnn = load_small_cnn_ascad_no_batch_norm()
        cnn.set_weights(best_indiv.weights)
        key_rank = exec_sca(cnn, x_test, y_test, ptexts_test, true_atk_subkey)

        avg_keyrank += key_rank/n_experiments

    return avg_keyrank


def single_ensemble_experiment():
    """
    Runs a single ensemble GA experiment with hard-coded parameters.
    """
    subkey_idx = 2
    (x_train, y_train, train_ptexts, target_train_subkey, x_atk, y_atk, \
        atk_ptexts, target_atk_subkey) = load_prepared_ascad_vars(
            subkey_idx=subkey_idx, scale=True, use_mlp=False, remote_loc=False
        )

    # Train the CNN by running it through the GA
    nn = load_small_cnn_ascad_no_batch_norm()
    # nn = load_small_mlp_ascad()

    pop_size = 50
    atk_set_size = 16
    select_fun = "tournament"
    exp_name = gen_experiment_name(pop_size, atk_set_size, select_fun)
    ga_results = run_ga(
        max_gens=100,
        pop_size=pop_size,
        mut_power=0.03,
        mut_rate=0.04,
        crossover_rate=0.5,
        mut_power_decay_rate=0.99,
        truncation_proportion=0.4,
        atk_set_size=atk_set_size,
        nn=nn,
        x_valid=x_train,
        y_valid=y_train,
        ptexts_valid=train_ptexts,
        x_test=x_atk,
        y_test=y_atk,
        ptexts_test=atk_ptexts,
        true_validation_subkey=target_train_subkey,
        true_atk_subkey=target_atk_subkey,
        subkey_idx=subkey_idx,
        parallelise=True,
        apply_fi=True,
        select_fun=select_fun,
        metric_type=MetricType.KEYRANK,
        experiment_name=exp_name,
        evaluate_on_test_set=False
    )

    ensemble_model_sca(
        ga_results, load_small_cnn_ascad_no_batch_norm, 10, x_atk, y_atk,
        target_atk_subkey, atk_ptexts, subkey_idx=subkey_idx,
        experiment_name=f"ensemble_{exp_name}"
    )


def ensemble_model_sca(ga_results, model_load_func, n_folds, x_atk, y_atk,
                       true_subkey, ptexts, subkey_idx=2,
                       experiment_name="ensemble_test"):
    """
    Evaluates an ensemble of the top performing neural networks from the given
    GA results on the given attack set over several folds. This is accomplished
    by using the bagging method, i.e. summing the prediction probabilities of
    each model.
    """
    top_indivs = ga_results[2]  # Sorted from best to worst fitness
    n_indivs = len(top_indivs)

    nns = np.empty(n_indivs, dtype=object)
    key_rankss = np.empty(n_indivs + 1, dtype=object)  # [[mean_key_rank]]
    # Extract NNs from GA results & perform SCAs for performance comparison
    for i in range(n_indivs):
        nns[i] = model_load_func()
        nns[i].set_weights(top_indivs[i].weights)

        key_rankss[i] = kfold_ascad_atk_with_varying_size(
            n_folds, nns[i], atk_data=(x_atk, y_atk, true_subkey, ptexts)
        )

    # Perform the ensemble SCA
    bagged_pred_probs = sum([nn.predict(x_atk) for nn in nns])
    ensemble_key_ranks = kfold_mean_key_ranks(
        bagged_pred_probs, ptexts, true_subkey, n_folds, subkey_idx, experiment_name
    )
    key_rankss[-1] = ensemble_key_ranks

    # Plot all lines in the same figure
    labels = [f"Top-{i + 1}" for i in range(10)] + ["Ensemble"]
    plot_n_traces_vs_key_rank(experiment_name, *key_rankss, labels=labels)

    print(f"Mean key rank with ensemble method: {ensemble_key_ranks[-1]}")


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
    cnn = build_small_cnn_ascad_trainable_conv()
    cnn.compile(optimizer, loss_fn)
    history = cnn.fit(x_train, y_train_converted, batch_size, n_epochs)

    # Save the model if desired
    if save:
        cnn.save('./trained_models/efficient_cnn_ascad_trained_conv.h5')

    # Attack with the trained model
    key_rank = exec_sca(cnn, x_atk_reshaped, y_atk, atk_ptexts, target_subkey, subkey_idx=2)

    print(f"Key rank obtained with efficient CNN on ASCAD: {key_rank}")


def attack_ascad_with_cnn(subkey_idx=2, atk_set_size=10000, scale=True):
    # Load attack set of 10k ASCAD traces and relevant metadata
    (x_atk, y_atk, target_subkey, atk_ptexts) = \
        load_ascad_atk_variables(for_cnns=True, subkey_idx=2, scale=scale)
    x_atk, y_atk = x_atk[:atk_set_size], y_atk[:atk_set_size]

    # Load CNN and attack the traces with it
    cnn = load_small_cnn_ascad(official=True)
    # cnn = build_small_cnn_ascad()
    
    # print(f"Keyrank = {exec_sca(cnn, x_atk, y_atk, atk_ptexts, target_subkey)}")
    kfold_ascad_atk_with_varying_size(
        10,
        cnn,
        subkey_idx=subkey_idx,
        experiment_name="test",
        atk_data=(x_atk, y_atk, target_subkey, atk_ptexts)
    )
    

def kfold_ascad_atk_with_varying_size(k, nn, subkey_idx=2, experiment_name="",
    atk_data=None):
    # Use the given data if possible. Load 10k ASCAD attack traces otherwise.
    (x_atk, y_atk, target_subkey, atk_ptexts) = \
        atk_data if atk_data \
        else load_ascad_atk_variables(for_cnns=True, subkey_idx=2, scale=True)

    # Predict outputs for the full set
    y_pred_probs = nn.predict(x_atk)

    mean_ranks = kfold_mean_key_ranks(
        y_pred_probs, atk_ptexts, target_subkey, k, subkey_idx, experiment_name
    )

    if experiment_name:
        plot_n_traces_vs_key_rank(experiment_name, mean_ranks)
    
    return mean_ranks


def compute_memory_requirements(pop_sizes, atk_set_sizes):
    """
    Approximates the required memory (GB) to run the GA for a given list of
    population sizes and attack set sizes.
    """
    cnn = load_small_cnn_ascad()

    print("=== Reqs based on theory: ===")
    for pop_size in pop_sizes:
        for atk_set_size in atk_set_sizes:
            mem = compute_mem_req(pop_size, cnn, atk_set_size)/1e6
            print(f"Required memory for pop size {pop_size} & attack set size {atk_set_size}: {mem} GB")

    print("==========================================")
    print("=== Reqs based on manual observations: ===")
    for pop_size in pop_sizes:
        for atk_set_size in atk_set_sizes:
            mem = compute_mem_req_from_known_vals(pop_size, atk_set_size)
            print(f"Required memory for pop size {pop_size} & attack set size {atk_set_size}: {mem} GB")

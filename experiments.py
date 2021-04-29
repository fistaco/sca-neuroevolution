import multiprocessing as mp
import os
import pickle
from time import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import METRIC_TYPE_MAP, SELECT_FUNCTION_MAP
from data_processing import (load_ascad_atk_variables, load_ascad_data,
                             load_prepared_ascad_vars, sample_data,
                             scale_inputs, shuffle_data, balanced_sample,
                             load_chipwhisperer_data)
from genetic_algorithm import (GeneticAlgorithm, evaluate_fitness,
                               train_nn_with_ga)
from helpers import (compute_fold_keyranks, compute_mem_req,
                     compute_mem_req_from_known_vals, exec_sca,
                     gen_experiment_name, gen_extended_exp_name,
                     gen_ga_grid_search_arg_lists, kfold_mean_key_ranks,
                     label_to_subkey)
from metrics import MetricType, keyrank
import models
from models import (build_small_cnn_ascad,
                    build_small_cnn_ascad_trainable_conv,
                    build_small_mlp_ascad,
                    build_small_mlp_ascad_trainable_first_layer,
                    load_nn_from_experiment_results, load_small_cnn_ascad,
                    load_small_cnn_ascad_no_batch_norm, load_small_mlp_ascad,
                    small_mlp_cw)
from nn_genome import NeuralNetworkGenome
from plotting import (plot_gens_vs_fitness, plot_n_traces_vs_key_rank,
                      plot_var_vs_key_rank, plot_2d, plot_3d)
from result_processing import ResultCategory, filter_df


def weight_evo_experiment_from_params(cline_args, remote=True):
    """
    Runs a weight evolution grid search experiment from a set of given
    parameters and stored the results for a given run index. This is useful
    for the repeating of experiment that failed to store results.
    """
    ga_params = [cline_args[i] for i in range(1, 9)]

    # Convert each param to its proper datatype and form
    ga_params[0] = int(ga_params[0])
    ga_params[1:5] = [float(param) for param in ga_params[1:5]]
    ga_params[5] = int(ga_params[5])
    ga_params[6] = SELECT_FUNCTION_MAP[ga_params[6]]
    ga_params[7] = METRIC_TYPE_MAP[ga_params[7]]
    run_idx = int(cline_args[-1])

    single_weight_evo_grid_search_experiment(
        exp_idx=777, run_idx=run_idx, params=tuple(ga_params), remote=remote
    )


def single_weight_evo_grid_search_experiment(exp_idx=0, run_idx=0,
                                             params=None, remote=True):
    """
    Executes an averaged GA experiment over 10 runs, where the arguments of the
    GA are determined by the given index for the generated list of GA
    argument tuples.
    """
    print(f"Starting experiment {exp_idx}/485...")

    # Load data
    subkey_idx = 2
    (x_train, y_train, train_ptexts, target_train_subkey, x_atk, y_atk, \
        atk_ptexts, target_atk_subkey) = load_prepared_ascad_vars(
            subkey_idx=subkey_idx, scale=True, use_mlp=True, remote_loc=remote
        )
    nn = models.NN_LOAD_FUNC()

    # Generate arguments based on the given experiment index
    (ps, mp, mr, mpdr, fdr, ass, sf, mt) = \
        params or gen_ga_grid_search_arg_lists()[exp_idx]
    exp_name = gen_extended_exp_name(ps, mp, mr, mpdr, fdr, ass, sf, mt, "mlp")

    # Disable parallelisation on the HPC cluster
    parallelise = not remote

    run_ga_for_grid_search(
        max_gens=50,
        pop_size=ps,
        mut_power=mp,
        mut_rate=mr,
        crossover_rate=0.5,
        mut_power_decay_rate=mpdr,
        truncation_proportion=0.4,
        atk_set_size=ass,
        nn=nn,
        x_valid=x_train,
        y_valid=y_train,
        ptexts_valid=train_ptexts,
        x_test=x_atk,
        y_test=y_atk,
        ptexts_test=atk_ptexts,
        true_validation_subkey=target_train_subkey,
        true_atk_subkey=target_atk_subkey,
        parallelise=parallelise,
        apply_fi=True,
        select_fun=sf,
        metric_type=mt,
        run_idx=run_idx,
        experiment_name=exp_name,
        save_results=True
    )

def run_ga_for_grid_search(max_gens, pop_size, mut_power, mut_rate,
           crossover_rate, mut_power_decay_rate, truncation_proportion,
           atk_set_size, nn, x_valid, y_valid, ptexts_valid, x_test, y_test,
           ptexts_test, true_validation_subkey, true_atk_subkey, parallelise,
           apply_fi, select_fun, metric_type, run_idx, experiment_name="test",
           save_results=True):
    """
    Runs a one GA experiment with the given parameters and stores the results
    in a directory specific to this experiment.
    """
    # Create results directory if necessary
    dir_path = f"results/{experiment_name}"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    print(f"Starting experiment {experiment_name} run {run_idx}.")
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

    # Create a new model from the best individual's weights and evaluate it
    nn = models.NN_LOAD_FUNC()
    nn.set_weights(best_indiv.weights)
    key_rank = exec_sca(nn, x_test, y_test, ptexts_test, true_atk_subkey)

    if save_results:
        (_, best_fitness_per_gen, top_ten) = ga.get_results()
        results = (best_indiv, best_fitness_per_gen, top_ten, key_rank)
        with open(f"{dir_path}/run{run_idx}_results.pickle", "wb") as f:
            pickle.dump(results, f)


def ga_grid_search_find_best_network():
    """
    Compare all of the given potential best networks by attacking a smaller
    data set.
    """
    # Extract best networks from results
    with open("res/ga_weight_evo_grid_key_rank_zero_indivs.pickle", "rb") as f:
        key_rank_zero_indivs = pickle.load(f)
    nns = np.empty(len(key_rank_zero_indivs), dtype=object)
    for i in range(len(nns)):
        nns[i] = models.NN_LOAD_FUNC()
        nns[i].set_weights(key_rank_zero_indivs[i][0].weights)

    # Load data
    subkey_idx = 2
    (_, _, _, _, x_atk, y_atk, atk_ptexts, target_atk_subkey) = \
        load_prepared_ascad_vars(subkey_idx, True, True, False)
    n_samples = 5000
    atk_set_size = len(x_atk)

    # Attack once with each NN
    nn_y_pred_probss = [nn.predict(x_atk) for nn in nns]
    avg_key_ranks = np.zeros(len(nns), dtype=float)

    # Prepare argument lists for parallel fold evaluation
    argss = [
        (i, np.random.choice(atk_set_size, n_samples), atk_ptexts, subkey_idx,
        atk_set_size, nn_y_pred_probss, target_atk_subkey, n_samples)
        for i in range(30)
    ]
    pool = mp.Pool(6)
    # For 30 folds, evaluate each NN's performance on the same subset of traces
    # The resulting array will have shape (30, len(nns))
    key_ranks_per_fold = pool.starmap(single_fold_multiple_nns_eval, argss)
    pool.close()
    pool.join()

    # Save intermediate results
    with open("res/ga_gs_best_nns_key_ranks_per_fold.pickle", "wb") as f:
        pickle.dump(key_ranks_per_fold, f)

    # Obtain the averages of the results
    for i in range(len(nns)):
        avg_key_ranks[i] = 0
        for fold in range(30):
            avg_key_ranks[i] += key_ranks_per_fold[fold][i]/30

    with open("res/ga_gs_best_networks_avg_key_ranks.pickle", "wb") as f:
        pickle.dump(avg_key_ranks, f)

    print(f"Best avg key rank: {np.min(avg_key_ranks)}")
    best_nn_idx = np.argmin(avg_key_ranks)
    with open("res/ga_gs_best_indiv.pickle", "wb") as f:
        # Save as (indiv, experiment_name)
        pickle.dump(key_rank_zero_indivs[best_nn_idx], f)


def single_fold_multiple_nns_eval(fold, indices, atk_ptexts, subkey_idx,
                                  atk_set_size, nn_y_pred_probss,
                                  true_subkey, n_samples):
    """
    Computes the key rank for each given list of individual NN predictions
    over a single fold of the given data set with the given indices.
    """
    ptexts = atk_ptexts[indices]

    avg_fold_key_ranks = np.zeros(len(nn_y_pred_probss), dtype=float)
    for (i, y_pred_probs) in enumerate(nn_y_pred_probss):
        run_idx = i*30 + fold
        pred_probs = y_pred_probs[indices]
        trace_amnt_key_ranks = compute_fold_keyranks(
            run_idx, pred_probs, ptexts, subkey_idx, atk_set_size, true_subkey
        )

        avg_fold_key_ranks[i] = trace_amnt_key_ranks[n_samples - 1]

    return avg_fold_key_ranks


def ga_grid_search_best_network_eval():
    """
    Attacks the ASCAD data set over 100 folds with the best NN resulting from
    the GA grid search and plots the results.
    """
    # Load best network
    with open("res/ga_gs_best_indiv.pickle", "rb") as f:
        # Save as (indiv, experiment_name)
        (indiv, exp_name) = pickle.load(f)
    nn = models.NN_LOAD_FUNC()
    nn.set_weights(indiv.weights)

    # Load data
    subkey_idx = 2
    (_, _, _, _, x_atk, y_atk, atk_ptexts, target_atk_subkey) = \
        load_prepared_ascad_vars(subkey_idx, True, True, False)
    n_folds = 100

    # Execute attack with the best NN over 100 folds
    kfold_ascad_atk_with_varying_size(
        n_folds,
        nn,
        subkey_idx=subkey_idx,
        experiment_name="best_grid_search_nn_100fold_eval",
        atk_data=(x_atk, y_atk, target_atk_subkey, atk_ptexts),
        parallelise=True
    )


def ensemble_atk_with_best_grid_search_networks(top_n=5):
    subkey_idx = 2
    (_, _, _, _, x_atk, y_atk, atk_ptexts, target_atk_subkey) = \
        load_prepared_ascad_vars(subkey_idx, True, True, False)

    # Load all best networks and find the top n
    with open("res/ga_weight_evo_grid_key_rank_zero_indivs.pickle", "rb") as f:
        key_rank_zero_indivs = pickle.load(f)
    with open("res/ga_gs_best_networks_avg_key_ranks.pickle", "rb") as f:
        avg_key_ranks = pickle.load(f)
    top_five_idxs = np.argsort(avg_key_ranks)[:top_n]
    nns = [models.NN_LOAD_FUNC() for i in range(top_n)]
    for i in range(top_n):
        indiv = key_rank_zero_indivs[top_five_idxs[i]][0]
        nns[i].set_weights(indiv.weights)

    ensemble_model_sca(
        nns, 30, x_atk, y_atk, target_atk_subkey, atk_ptexts, subkey_idx,
        "ensemble_from_best_grid_search_networks"
    )


def ga_grid_search_parameter_influence_eval():
    # Load df and params for best average performance
    with open("res/ga_weight_evo_grid_search_results.pickle", "rb") as f:
        df = pickle.load(f)
    with open("res/ga_weight_evo_grid_best_experiment_data.pickle", "rb") as f:
        (ps, mp, mr, mpdr, fdr, ass, sm, _, _, _, key_rank) = pickle.load(f)
    params = (ps, mp, mr, mpdr, fdr, ass, sm)

    # For each variable, plot its influence on the final key rank
    result_categories = list(ResultCategory)
    for result_cat in result_categories[:7]:
        sub_df = filter_df(df, params, exempt_idx=result_cat.value)
        plot_var_vs_key_rank(
            sub_df[:, result_cat.value],
            sub_df[:, ResultCategory.KEY_RANK.value],
            result_cat
        )


def run_ga(max_gens, pop_size, mut_power, mut_rate, crossover_rate,
           mut_power_decay_rate, truncation_proportion, atk_set_size, nn,
           x_valid, y_valid, ptexts_valid, x_test, y_test, ptexts_test,
           true_validation_subkey, true_atk_subkey, subkey_idx, parallelise,
           apply_fi, select_fun, metric_type, n_atk_folds=1,
           experiment_name="test", remote=False, evaluate_on_test_set=True):
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
        metric_type,
        n_atk_folds=n_atk_folds,
        remote=remote
    )

    # Obtain the best network resulting from the GA
    start = time()
    best_indiv = \
        ga.run(nn, x_valid, y_valid, ptexts_valid, true_validation_subkey, subkey_idx)
    end = time()
    t = int(end-start)
    print(f"Time elapsed: {t}")

    # Save and plot results
    ga_results = ga.save_results(best_indiv, experiment_name)
    plot_gens_vs_fitness(experiment_name, ga.best_fitness_per_gen)

    if evaluate_on_test_set:
        # Create a new model from the best individual's weights and test it
        nn.set_weights(best_indiv.weights)
        kfold_ascad_atk_with_varying_size(
            1,
            nn,
            2,
            experiment_name,
            atk_data=(x_test, y_test, true_atk_subkey, ptexts_test),
            parallelise=parallelise,
            remote=remote
        )

        # Evaluate the best network's performance on the test set
        key_rank = exec_sca(nn, x_test, y_test, ptexts_test, true_atk_subkey)
        print(f"Key rank on test set with exec_sca: {key_rank}")

    print(f"Key rank on validation set: {best_indiv.fitness}")
    
    return ga_results


def single_ga_experiment(remote_loc=False, use_mlp=False, averaged=False,
                         apply_fi=False, parallelise=False):
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
    nn = models.NN_LOAD_FUNC()

    pop_size = 52
    atk_set_size = 1024
    select_fun = "tournament"
    execution_func = run_ga if not averaged else averaged_ga_experiment
    execution_func(
        max_gens=25,
        pop_size=pop_size,
        mut_power=0.03,
        mut_rate=0.04,
        crossover_rate=0.0,
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
        parallelise=parallelise,
        apply_fi=apply_fi,
        select_fun=select_fun,
        metric_type=MetricType.INCREMENTAL_KEYRANK,
        n_atk_folds=10,
        experiment_name=gen_experiment_name(pop_size, atk_set_size, select_fun) + "multifold_test",
        remote=remote_loc
    )


def averaged_ga_experiment(max_gens, pop_size, mut_power, mut_rate,
           crossover_rate, mut_power_decay_rate, truncation_proportion,
           atk_set_size, nn, x_valid, y_valid, ptexts_valid, x_test, y_test,
           ptexts_test, true_validation_subkey, true_atk_subkey, parallelise,
           apply_fi, select_fun, metric_type, n_atk_folds=1, subkey_idx=2,
           n_experiments=10, experiment_name="test", remote=False,
           save_results=True):
    """
    Runs a given amount of GA experiments with the given parameters and returns
    the average key rank obtained with the full given attack set.

    The results of each run are stored in a directory specific to this
    experiment.
    """
    # Create results directory if necessary
    dir_path = f"results/{experiment_name}"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    avg_keyrank = 0
    for i in range(n_experiments):
        print(f"Starting experiment {experiment_name} run {i}.")
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
            metric_type,
            n_atk_folds=n_atk_folds,
            remote=remote
        )

        best_indiv = \
            ga.run(nn, x_valid, y_valid, ptexts_valid, true_validation_subkey,
                   subkey_i=subkey_idx)

        # Create a new model from the best individual's weights and evaluate it
        cnn = models.NN_LOAD_FUNC()
        cnn.set_weights(best_indiv.weights)
        key_rank = exec_sca(cnn, x_test, y_test, ptexts_test, true_atk_subkey)

        if save_results:
            (_, best_fitness_per_gen, top_ten) = ga.get_results()
            results = (best_indiv, best_fitness_per_gen, top_ten, key_rank)
            with open(f"{dir_path}/run{i}_results.pickle", "wb") as f:
                pickle.dump(results, f)

        avg_keyrank += key_rank/n_experiments

    return avg_keyrank


def single_ensemble_experiment():
    """
    Runs a single ensemble GA experiment with hard coded parameters.
    """
    subkey_idx = 2
    (x_train, y_train, train_ptexts, target_train_subkey, x_atk, y_atk, \
        atk_ptexts, target_atk_subkey) = load_prepared_ascad_vars(
            subkey_idx=subkey_idx, scale=True, use_mlp=False, remote_loc=False
        )

    # Train the CNN by running it through the GA
    model_load_func = load_small_mlp_ascad
    nn = model_load_func()

    pop_size = 50
    atk_set_size = 16
    select_fun = "tournament"
    exp_name = gen_experiment_name(pop_size, atk_set_size, select_fun)
    ga_results = run_ga(
        max_gens=50,
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
        metric_type=MetricType.KEYRANK_AND_ACCURACY,
        experiment_name=exp_name,
        evaluate_on_test_set=False
    )

    top_indivs = ga_results[2]  # Sorted from best to worst fitness
    n_indivs = len(top_indivs)

    # Extract NNs from GA results
    nns = np.empty(n_indivs, dtype=object)
    for i in range(n_indivs):
        nns[i] = models.NN_LOAD_FUNC()
        nns[i].set_weights(top_indivs[i].weights)

    ensemble_model_sca(
        nns, 10, x_atk, y_atk, target_atk_subkey, atk_ptexts,
        subkey_idx=subkey_idx, experiment_name=f"ensemble_{exp_name}"
    )


def ensemble_model_sca(nns, n_folds, x_atk, y_atk, true_subkey, ptexts,
                       subkey_idx=2, experiment_name="ensemble_test"):
    """
    Evaluates an ensemble of the given top performing neural networks on the
    given attack set over several folds. This is accomplished by using the
    bagging method, i.e. summing the prediction probabilities of each model.
    """
    print("Performing SCAs with individual networks for comparison...")
    # Perform SCAs for comparison for the 1st, 3rd, and 5th best NNs
    nn_indices = [0, 2, 4]
    key_rankss = np.empty(len(nn_indices) + 1, dtype=object)  # [[key_rank]]
    for (i, nn_idx) in enumerate(nn_indices):
        key_rankss[i] = kfold_ascad_atk_with_varying_size(
            n_folds, nns[nn_idx], atk_data=(x_atk, y_atk, true_subkey, ptexts),
            parallelise=True
        )

    print("Commencing ensemble attack.")
    # Perform the ensemble SCA
    bagged_pred_probs = sum([nn.predict(x_atk) for nn in nns])
    ensemble_key_ranks = kfold_mean_key_ranks(
        bagged_pred_probs, ptexts, true_subkey, n_folds, subkey_idx,
        experiment_name, parallelise=True
    )
    key_rankss[-1] = ensemble_key_ranks

    # Plot all lines in the same figure
    labels = [f"Top-{i + 1}" for i in nn_indices] + ["Ensemble"]
    plot_n_traces_vs_key_rank(experiment_name, *key_rankss, labels=labels)

    print(f"Mean key rank with ensemble method: {ensemble_key_ranks[-1]}")


def small_cnn_sgd_sca(save=False, subkey_idx=2):
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
        cnn.save('./trained_models/efficient_cnn_ascad.h5')

    # Attack with the trained model
    key_rank = exec_sca(cnn, x_atk_reshaped, y_atk, atk_ptexts, target_subkey, subkey_idx)

    print(f"Key rank obtained with efficient CNN on ASCAD: {key_rank}")


def train_first_layer_ascad_mlp():
    """
    Trains the first layer of the efficient ASCAD MLP model and stores it.
    """
    subkey_idx = 2
    (x_train, y_train, _, _, x_atk, y_atk, atk_ptexts, target_atk_subkey) = \
        load_prepared_ascad_vars(
            subkey_idx=subkey_idx, scale=True, use_mlp=True, remote_loc=False,
            for_sgd=True
        )

    # Define hyperparameters
    n_epochs = 100
    batch_size = 50
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

    # Train
    nn = build_small_mlp_ascad_trainable_first_layer(save=False)
    nn.compile(optimizer, loss_fn)
    history = nn.fit(x_train, y_train, batch_size, n_epochs)

    nn.save("./trained_models/efficient_mlp_ascad_model_trained_first.h5")

    # Attack with the trained model
    key_rank = exec_sca(nn, x_atk, y_atk, atk_ptexts, target_atk_subkey, subkey_idx)
    print(f"Key rank obtained after training the first FC layer: {key_rank}")


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
        3,
        cnn,
        subkey_idx=subkey_idx,
        experiment_name="test",
        atk_data=(x_atk, y_atk, target_subkey, atk_ptexts),
        parallelise=False
    )


def attack_chipwhisperer_mlp(subkey_idx=1, save=False, train_with_ga=True,
                             remote=False, ass=256, folds=5,
                             select_fn="roulette_wheel"):
    (x_train, y_train, pt_train, x_atk, y_atk, pt_atk, k) = \
        load_chipwhisperer_data(n_train=8000, subkey_idx=1, remote=remote)

    suffix = "_ga" if train_with_ga else "_sgd"
    exp_name = f"chipwhisperer_mlp_{suffix}_test"

    # Load and train MLP
    nn = None
    if train_with_ga:
        nn = small_mlp_cw(build=False)
        nn = train_nn_with_ga(
            nn, x_train, y_train, pt_train, k, subkey_idx, atk_set_size=ass,
            select_fn=select_fn, metric_type=MetricType.KEYRANK_PROGRESS,
            parallelise=True, shuffle_traces=False, n_atk_folds=folds,
            remote=remote, t_size=3, max_gens=5, pop_size=50,
            crossover_rate=0.2, plot_fit_progress=True, exp_name=exp_name,
            debug=True, truncation_proportion=0.4, mut_power=0.07,
            mut_rate=0.07
        )
    else:
        nn = small_mlp_cw(build=True)
        y_train = keras.utils.to_categorical(y_train)
        n_epochs = 50
        batch_size = 50
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
        nn.compile(optimizer, loss_fn)
        history = nn.fit(x_train, y_train, batch_size, n_epochs)
    if save:
        nn.save(f"./trained_models/cw_mlp_trained{suffix}.h5")

    kfold_ascad_atk_with_varying_size(
        26,
        nn,
        subkey_idx=subkey_idx,
        experiment_name=exp_name,
        atk_data=(x_atk, y_atk, k, pt_atk),
        parallelise=True
    )


def kfold_ascad_atk_with_varying_size(k, nn, subkey_idx=2, experiment_name="",
    atk_data=None, parallelise=False, remote=False):
    # Use the given data if possible. Load 10k ASCAD attack traces otherwise.
    (x_atk, y_atk, target_subkey, atk_ptexts) = \
        atk_data if atk_data \
        else load_ascad_atk_variables(for_cnns=True, subkey_idx=2, scale=True)

    # Predict outputs for the full set
    y_pred_probs = nn.predict(x_atk)

    mean_ranks = kfold_mean_key_ranks(
        y_pred_probs, atk_ptexts, target_subkey, k, subkey_idx,
        experiment_name, parallelise=parallelise, remote=remote
    )

    if experiment_name:
        plot_n_traces_vs_key_rank(experiment_name, mean_ranks)
    
    return mean_ranks


def test_fitness_function_consistency(nn_quality="medium"):
    """
    Determines the consistency of a several metric types by computing fitness
    evaluation standard deviations with different amounts of traces and folds
    for multiple random neural networks.
    """
    np.random.seed(77)

    n_indivs = 10
    # Load networks according to desired network quality
    if nn_quality == "low":  # completely random networks
        base_weights = models.NN_LOAD_FUNC().get_weights()
        indivs = [NeuralNetworkGenome(base_weights) for _ in range(n_indivs)]
        for indiv in indivs:
            indiv.random_weight_init()
    if nn_quality == "medium":  # best networks from a prior random search
        indivs_path = "res/ga_weight_evo_grid_key_rank_zero_indivs.pickle"
        with open(indivs_path, "rb") as f:
            key_rank_zero_indivs = np.array(pickle.load(f), dtype=object)
        with open("res/ga_gs_best_networks_avg_key_ranks.pickle", "rb") as f:
            avg_key_ranks = pickle.load(f)
        top_idxs = np.argsort(avg_key_ranks)[:n_indivs]
        indivs = [tup[0] for tup in key_rank_zero_indivs[top_idxs]]
        del key_rank_zero_indivs

    # Load data
    subkey_idx = 2
    (x_train, y_train, pt_train, k_train, x_atk, y_atk, pt_atk, k_atk) = \
        load_prepared_ascad_vars(
            subkey_idx=subkey_idx, scale=True, use_mlp=True, remote_loc=False
        )

    # Set up DF
    trace_amnts = [256, 512, 768, 1024]
    fold_amnts = [5, 10, 15, 20]
    metric_types = [MetricType.KEYRANK, MetricType.INCREMENTAL_KEYRANK,
                    MetricType.KEYRANK_PROGRESS]
    n_evals = n_indivs*len(trace_amnts)*len(fold_amnts)*len(metric_types)*10
    df = np.zeros(n_evals, dtype=[
        ("n_traces", np.uint16), ("n_folds", np.uint8), ("metric", MetricType),
        ("indiv_id", np.uint8), ("fitness", np.float32)
    ])

    pool = mp.Pool(5)

    # Generate data samples and evaluate individuals
    i = 0
    for t in trace_amnts:
        for f in fold_amnts:
            for _ in range(10):  # Repeat everything 10 times
                sets = [
                    balanced_sample(t, x_train, y_train, pt_train, 256, True)
                    for _ in range(f)
                ]

                for m in metric_types:
                    for (indiv_id, indiv) in enumerate(indivs):
                        print(f"Running evaluation {i}/{n_evals - 1}.")
                        argss = [
                            (indiv.weights, s[0], s[1], s[2], k_train,
                                subkey_idx, m, t)
                            for s in sets
                        ]
                        fitnesses = pool.starmap(evaluate_fitness, argss)
                        fitness = np.mean(fitnesses)

                        df[i] = (t, f, m, indiv_id, fitness)
                        i += 1
    print("Finished.")

    df_path = f"res/fitness_consistency_eval_df_{nn_quality}.pickle"
    with open(df_path, "wb") as f:
        pickle.dump(df, f)

    pool.close()
    pool.join()

    # For each m-t-f combination, store the mean of all individuals' stdevs
    mean_stdevs = {m: { t: { f:
        np.mean(
            [
                np.std(df[
                    (df["n_traces"] == t) & (df["n_folds"] == f) & \
                    (df["metric"] == m) & (df["indiv_id"] == i)
                ]["fitness"])
                for i in range(n_indivs)
            ]
        )
    for f in fold_amnts} for t in trace_amnts} for m in metric_types}

    fs_stds_path = f"res/fitness_consistency_mean_stds_{nn_quality}.pickle"
    with open(fs_stds_path, "wb") as f:
        pickle.dump(mean_stdevs, f)

    # # For each metric, plot n_traces & n_folds vs. mean fitness std.
    for m in metric_types:
        zs = np.zeros((len(trace_amnts),  len(fold_amnts)), dtype=np.float64)
        for (i, n_traces) in enumerate(trace_amnts):
            for (j, n_folds) in enumerate(fold_amnts):
                zs[i, j] = mean_stdevs[m][n_traces][n_folds]

        plot_3d(
            fold_amnts, trace_amnts, zs,
            "Folds", "Traces", "Std. dev.",
            f"Folds & traces ~ fitness standard deviation ({m.name})"
        )


def test_inc_kr_fold_consistency(nn_quality="medium"):
    """
    Determines the consistency of the incremental key rank metric by computing
    its standard deviation for different amounts of folds of 256 balanced
    traces, averaged over 10 medium quality individuals.
    """
    np.random.seed(77)

    # Load networks
    n_indivs = 10
    indivs_path = "res/ga_weight_evo_grid_key_rank_zero_indivs.pickle"
    with open(indivs_path, "rb") as f:
        key_rank_zero_indivs = np.array(pickle.load(f), dtype=object)
    with open("res/ga_gs_best_networks_avg_key_ranks.pickle", "rb") as f:
        avg_key_ranks = pickle.load(f)
    top_idxs = np.argsort(avg_key_ranks)[:n_indivs]
    indivs = [tup[0] for tup in key_rank_zero_indivs[top_idxs]]
    del key_rank_zero_indivs

    # Load data
    subkey_idx = 2
    (x_train, y_train, pt_train, k_train, x_atk, y_atk, pt_atk, k_atk) = \
        load_prepared_ascad_vars(
            subkey_idx=subkey_idx, scale=True, use_mlp=True, remote_loc=False
        )

    # Set up DF
    m = MetricType.INCREMENTAL_KEYRANK
    t = 256
    fold_amnts = [25, 30, 35, 40, 45, 50]
    n_evals = n_indivs*len(fold_amnts)*10
    df = np.zeros(n_evals, dtype=[
        ("n_folds", np.uint8), ("indiv_id", np.uint8), ("fitness", np.float32)
    ])

    pool = mp.Pool(5)

    i = 0
    for f in fold_amnts:
        for _ in range(10):  # Repeat everything 10 times
            sets = [
                balanced_sample(t, x_train, y_train, pt_train, 256, True)
                for _ in range(f)
            ]

            for (indiv_id, indiv) in enumerate(indivs):
                print(f"Running evaluation {i}/{n_evals - 1}.")
                argss = [
                    (indiv.weights, s[0],s[1],s[2], k_train, subkey_idx, m, t)
                    for s in sets
                ]
                fitnesses = pool.starmap(evaluate_fitness, argss)
                fitness = np.mean(fitnesses)

                df[i] = (f, indiv_id, fitness)
                i += 1
    print("Finished.")

    df_path = f"res/inc_kr_fold_consistency_df_{nn_quality}.pickle"
    with open(df_path, "wb") as f:
        pickle.dump(df, f)

    pool.close()
    pool.join()

    mean_stdevs = { f:
        np.mean(
            [
                np.std(df[
                    (df["n_folds"] == f) & (df["indiv_id"] == i)
                ]["fitness"])
                for i in range(n_indivs)
            ]
        )
    for f in fold_amnts}

    fs_stds_path = f"res/inc_kr_fold_consistency_mean_stds_{nn_quality}.pickle"
    with open(fs_stds_path, "wb") as f:
        pickle.dump(mean_stdevs, f)

    # Plot n_folds vs. mean fitness std.
    ys = [mean_stdevs[f] for f in fold_amnts]

    plot_2d(fold_amnts, ys, "Folds", "Std. dev.",
            f"Folds ~ fitness standard deviation ({m.name})"
    )


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

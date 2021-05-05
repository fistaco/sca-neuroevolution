import pickle
from enum import Enum
import sys

import numpy as np
from helpers import gen_extended_exp_name, gen_mini_grid_search_arg_lists
from nn_genome import NeuralNetworkGenome


def combine_grid_search_results():
    """
    Loads the experimental results from experiments that were run for a grid
    search using the averaged_ga_experiment method.
    """
    argss = gen_mini_grid_search_arg_lists()
    n_exps = len(argss)

    best_inc_kr = 3.0
    best_run_results = None
    best_avg_inc_kr = 3.0
    best_experiment_data = None
    n_repeats = 5

    df = np.zeros((n_exps*n_repeats, len(ResultCategory)), dtype=object)
    run_nr = 0
    for (exp_idx, args) in enumerate(argss):
        (ps, mp, mr, mpdr, fdr, ass, sf, mt, n_folds, fi, bt, tp, cor) = args
        print(f"Processing results from experiment {exp_idx}/{n_exps - 1}")

        name = gen_extended_exp_name(*args)
        dir_path = f"grid-search-results/{name}"

        avg_inc_kr = 0
        for i in range(n_repeats):
            # Load results
            filepath = f"{dir_path}/run{i}_results.pickle"
            results = None
            with open(filepath, "rb") as f:
                results = pickle.load(f)
            (best_indiv, best_fitness_per_gen, top_ten, fit, inc_kr) = results

            avg_inc_kr += inc_kr/n_repeats

            # # Update best results so far
            if inc_kr < best_inc_kr:
                best_inc_kr = inc_kr
                best_run_results = results

                print(f"=== New best INC_KR: {inc_kr} ===")
                print(f"=== Achieved with: {name} ===")

            # Put results in dataframe
            df[run_nr] = [mp, mr, mpdr, sf, tp, cor, fit, inc_kr]

            run_nr += 1

        # Update best results so far
        if avg_inc_kr < best_avg_inc_kr:
            best_avg_inc_kr = avg_inc_kr
            best_experiment_data = (mp, mr, mpdr, sf, tp, cor, exp_idx, inc_kr)

            print(f"=== New best avg key rank: {avg_inc_kr} ===")
            print(f"=== Achieved with: {name} ===")

    corr_mat = np.corrcoef(df[:, [-1, -2]].astype(np.float64).T)
    print(f"Correlation between final INC_KR and fitness: {corr_mat[0, 1]}")

    with open("res/static_gs_weight_evo_results_df.pickle", "wb") as f:
        pickle.dump(df, f)
    with open("res/static_gs_weight_evo_best_run_results.pickle", "wb") as f:
        pickle.dump(best_run_results, f)
    with open("res/static_gs_weight_evo_best_exp_data.pickle", "wb") as f:
        pickle.dump(best_experiment_data, f)

    print("Finished processing and saving grid search results.")


def filter_df(df, variables, exempt_idx=-1):
    """
    Returns a filtered data frame (DF) containing the DF rows of which the
    values in the columns correspond to the given variables, with the
    exemption of the variable of a given exempt index.
    """
    df_sub = df
    for (i, var) in enumerate(variables):
        if i == exempt_idx:
            continue
        df_sub = df_sub[df_sub[:, i] == variables[i]]

    return df_sub


def load_fitness_vs_keyrank_results_df(exp_name, n_experiments=10):
    """
    Constructs a dataframe containing the key rank and fitness value form GA
    run results of multiple experiments with parameters according to the given
    experiment name. 
    """
    df = np.zeros((n_experiments, 2), dtype=np.float32)
    dir_path = f"results/{exp_name}"

    for i in range(n_experiments):
        with open(f"{dir_path}/run{i}_results.pickle", "rb") as f:
            (best_indiv, _, _, key_rank) = pickle.load(f)
        df[i] = [best_indiv.fitness, key_rank]

    return df


def fitness_keyrank_corr(df, fit_col_idx=0, kr_col_idx=1):
    """
    Computes and returns the correlation between the fitness and key rank
    columns in the given DataFrame (DF). If no column indices are given, the
    DF should solely have a fitness column and a keyrank column. 
    """
    return np.corrcoef(df[:, [fit_col_idx, kr_col_idx]].T)[0, 1]


class ResultCategory(Enum):
    POPULATION_SIZE = 0
    MUTATION_POWER = 1
    MUTATION_RATE = 2
    MUTATION_POWER_DECAY_RATE = 3
    FITNESS_INHERITANCE_DECAY_RATE = 4
    ATTACK_SET_SIZE = 5
    SELECTION_METHOD = 6
    METRIC_TYPE = 7
    NETWORK_TYPE = 8
    KEY_RANK = 9
    FITNESS = 10

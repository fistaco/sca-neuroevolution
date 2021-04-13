import pickle
from enum import Enum
import sys

import numpy as np
from helpers import gen_extended_exp_name, gen_ga_grid_search_arg_lists
from nn_genome import NeuralNetworkGenome


def combine_grid_search_results():
    """
    Loads the experimental results from experiments that were run for a grid
    search using the averaged_ga_experiment method.
    """
    argss = gen_ga_grid_search_arg_lists()
    n_exps = len(argss)

    key_rank_zero_indivs = []

    best_key_rank = 255
    best_run_results = None
    best_avg_keyrank = 255
    best_experiment_data = None

    df = np.zeros((n_exps*10, len(ResultCategory)), dtype=object)
    run_nr = 0
    for (exp_idx, (ps, mp, mr, mpdr, fdr, ass, sm, mt)) in enumerate(argss):
        sys.stdout.write(f"\rProcessing results from experiment {exp_idx}/{n_exps - 1}")
        sys.stdout.flush()

        name = gen_extended_exp_name(ps, mp, mr, mpdr, fdr, ass, sm, mt, "mlp")
        dir_path = f"grid-search-results/{name}"

        avg_key_rank = 0
        for i in range(10):
            # Load results
            filepath = f"{dir_path}/run{i}_results.pickle"
            results = None
            with open(filepath, "rb") as f:
                results = pickle.load(f)
            (best_indiv, _, _, key_rank) = results
            fitness = best_indiv.fitness

            avg_key_rank += key_rank/10

            # # Update best results so far
            if key_rank < best_key_rank:
                best_key_rank = key_rank
                best_run_results = results

                print(f"=== New best key rank: {key_rank} ===")
                print(f"=== Achieved with: {name} ===")
            
            # Store the potential best NNs
            if key_rank == 0:
                key_rank_zero_indivs.append((best_indiv, name))

            # Put results in dataframe
            df[run_nr] = \
                [ps, mp, mr, mpdr, fdr, ass, sm, mt, "mlp", key_rank, fitness]

            run_nr += 1

        # Update best results so far
        if avg_key_rank < best_avg_keyrank:
            best_avg_keyrank = avg_key_rank
            best_experiment_data = \
                (ps, mp, mr, mpdr, fdr, ass, sm, mt, "mlp", exp_idx, key_rank)

            print(f"=== New best avg key rank: {avg_key_rank} ===")
            print(f"=== Achieved with: {name} ===")
    
    cols = [ResultCategory.KEY_RANK.value, ResultCategory.FITNESS.value]
    corr_mat = np.corrcoef(df[:, cols].astype(np.float64).T)
    print(f"Correlation between key rank and fitness: {corr_mat[0, 1]}")
    print(f"\nAmount of NNs with key rank 0: {len(key_rank_zero_indivs)}")

    with open("ga_weight_evo_grid_search_results.pickle", "wb") as f:
        pickle.dump(df, f)
    with open("ga_weight_evo_grid_search_best_run_results.pickle", "wb") as f:
        pickle.dump(best_run_results, f)
    with open("ga_weight_evo_grid_best_experiment_data.pickle", "wb") as f:
        pickle.dump(best_experiment_data, f)
    with open("ga_weight_evo_grid_key_rank_zero_indivs.pickle", "wb") as f:
        pickle.dump(key_rank_zero_indivs, f)

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

import multiprocessing as mp
from time import time
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # Hides warnings when using small data sets
from tensorflow import keras

from data_processing import load_ascad_data, train_test_split
from experiments import (attack_ascad_with_cnn, compute_memory_requirements,
                         run_ga, single_ensemble_experiment,
                         single_ga_experiment,
                         single_weight_evo_grid_search_experiment,
                         weight_evo_experiment_from_params,
                         small_cnn_sgd_sca, train_first_layer_ascad_mlp,
                         ga_grid_search_parameter_influence_eval)
from params import (ATTACK_SET_SIZE, CROSSOVER_RATE, FITNESS_INHERITANCE_DECAY,
                    MAX_GENERATIONS, MUTATION_POWER, MUTATION_POWER_DECAY,
                    MUTATION_RATE, POPULATION_SIZE, TOURNAMENT_SIZE,
                    TRUNCATION_PROPORTION)
from result_processing import combine_grid_search_results


def dual_parallel_weight_evo_experiment(args, remote=True):
    pool = mp.Pool(2)

    res = pool.apply_async(single_weight_evo_grid_search_experiment, (int(args[1]), int(args[2]), None, remote))
    res = pool.apply_async(single_weight_evo_grid_search_experiment, (int(args[1]), int(args[2]) + 1, None, remote))

    pool.close()
    pool.join()


if __name__ == "__main__":
    # dual_parallel_weight_evo_experiment(sys.argv, remote=False)
    # weight_evo_experiment_from_params(sys.argv, remote=True)
    # combine_grid_search_results()

    # train_first_layer_ascad_mlp()
    single_ga_experiment(remote_loc=False, use_mlp=True, averaged=False, apply_fi=False)
    # single_ensemble_experiment()

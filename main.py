import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # Hides warnings when using small data sets
from tensorflow import keras

from data_processing import load_ascad_data, train_test_split
from experiments import (attack_ascad_with_cnn, compute_memory_requirements,
                         ga_grid_search, run_ga, single_ga_experiment,
                         small_cnn_sgd_sca, single_ensemble_experiment,
                         single_weight_evo_grid_search_experiment)
from params import (ATTACK_SET_SIZE, CROSSOVER_RATE, FITNESS_INHERITANCE_DECAY,
                    MAX_GENERATIONS, MUTATION_POWER, MUTATION_POWER_DECAY,
                    MUTATION_RATE, POPULATION_SIZE, TOURNAMENT_SIZE,
                    TRUNCATION_PROPORTION)


def setup_tf_gpu_parallelism():
    pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Missing arguments: experiment_idx, run_idx")
    single_weight_evo_grid_search_experiment(int(sys.argv[1]), int(sys.argv[2]))
    # single_ga_experiment(remote_loc=False, use_mlp=False)
    # single_ensemble_experiment()
    # small_cnn_sgd_sca(subkey_idx=1)
    # attack_ascad_with_cnn(subkey_idx=2, atk_set_size=1000, scale=True)

    # pop_sizes = list(range(25, 251, 75))
    # pop_sizes = [50, 64]
    # atk_set_sizes = [2, 16, 128, 1024]
    # compute_memory_requirements(pop_sizes, atk_set_sizes)

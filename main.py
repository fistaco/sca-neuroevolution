import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras

from data_processing import load_ascad_data, train_test_split
from experiments import ga_grid_search, run_ga, small_cnn_sgd_sca, \
    attack_ascad_with_cnn, compute_memory_requirements, single_ga_experiment
from models import build_small_cnn_ascad
from params import MUTATION_POWER, MUTATION_RATE, CROSSOVER_RATE, \
    MAX_GENERATIONS, POPULATION_SIZE, TRUNCATION_PROPORTION, TOURNAMENT_SIZE, \
    MUTATION_POWER_DECAY, FITNESS_INHERITANCE_DECAY, ATTACK_SET_SIZE


if __name__ == "__main__":
    # single_ga_experiment()
    # small_cnn_sgd_sca(subkey_idx=2)
    attack_ascad_with_cnn(subkey_idx=2, atk_set_size=1000, scale=True)

    # pop_sizes = list(range(25, 251, 75))
    # atk_set_sizes = [2, 16, 128, 1024]
    # compute_memory_requirements(pop_sizes, atk_set_sizes)

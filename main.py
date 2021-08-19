import multiprocessing as mp

from numpy import random

mp.set_start_method("spawn", force=True)
from time import time
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

tf.get_logger().setLevel("ERROR")  # Hides warnings when using small data sets
from tensorflow import keras

from data_processing import load_ascad_data, train_test_split
from experiments import (attack_ascad_with_cnn, compute_memory_requirements,
                         run_ga, single_ensemble_experiment,
                         single_ga_experiment,
                         single_weight_evo_grid_search_experiment,
                         weight_evo_experiment_from_params,
                         small_cnn_sgd_sca, train_first_layer_ascad_mlp,
                         ga_grid_search_parameter_influence_eval,
                         test_inc_kr_fold_consistency,
                         attack_chipwhisperer_mlp, neat_experiment,
                         weight_evo_results_from_exp_names,
                         eval_best_nn_from_exp_name, train_and_attack_ascad,
                         build_small_mlp_ascad,
                         train_and_attack_with_multiple_nns,
                         weight_heatmaps_from_exp_name)
from metrics import MetricType
from models import (build_single_hidden_layer_mlp_ascad, mini_mlp_cw,
                    set_nn_load_func, train)
from neat_sca import set_global_data
from params import *
from result_processing import combine_grid_search_results

# Encode configurations as (max_gens, hw, avg_pooling, dataset_name, only_evolve_hidden, noise, desync, fs_neat)
configs = [
    # AP/HW combinations for full NEAT evolution
    (250, False, False, "ascad", False, 0, 0, False),
    (250, False, True,  "ascad", False, 0, 0, False),
    (250, True,  False, "ascad", False, 0, 0, False),
    (250, True,  True,  "ascad", False, 0, 0, False),
    # AP/HW combinations for hidden-only NEAT evolution
    (250, False, False, "ascad", True, 0, 0, False),
    (250, False, True,  "ascad", True, 0, 0, False),
    (250, True,  False, "ascad", True, 0, 0, False),
    (250, True,  True,  "ascad", True, 0, 0, False),
    # NEAT vs. CW with countermeasures -> only perform with AP + HW & hidden-only evolution
    (250, True,  True,  "cw", True, 0.05, 0,   False),
    (250, True,  True,  "cw", True, 0,    50,  False),
    (250, True,  True,  "cw", True, 0,    100, False),
    (250, True,  True,  "cw", True, 0.05, 50,  False),
    # Full FS-NEAT vs. unprotected CW
    (1000, True,  True,  "cw", False, 0, 0, True)
]
cf = configs[int(sys.argv[1])]
k_idx = 1 if cf[3] == "cw" else 2
pool_param = 4 if cf[3] == "cw" else 2

# set_nn_load_func("mini_mlp_cw")
# set_nn_load_func("mlp_cw", (False, False, 1))
# set_nn_load_func("mlp_ascad")

# set_global_data(
#     "ascad", 3584, subkey_idx=2, n_folds=1, remote=True, hw=bool(int(sys.argv[1])),
#     metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY, balanced=True,
#     use_sgd=True, use_avg_pooling=bool(int(sys.argv[2])), seed=77,
#     pool_param=2
# )
set_global_data(
    cf[3], 3584, subkey_idx=k_idx, n_folds=1, remote=False, hw=cf[1],
    metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY, balanced=True,
    use_sgd=True, use_avg_pooling=cf[2], seed=77, pool_param=pool_param,
    balance_on_hw=False
)


def dual_parallel_weight_evo_experiment(args, remote=True):
    pool = mp.Pool(2)

    res = pool.apply_async(single_weight_evo_grid_search_experiment, (int(args[1]), int(args[2]), None, remote))
    res = pool.apply_async(single_weight_evo_grid_search_experiment, (int(args[1]), int(args[2]) + 1, None, remote))

    pool.close()
    pool.join()


if __name__ == "__main__":
    # custom_params = (POPULATION_SIZE, MUTATION_POWER, MUTATION_RATE, MUTATION_POWER_DECAY, FITNESS_INHERITANCE_DECAY, 3584,
    #                  SELECTION_FUNCTION, METRIC_TYPE, 1, APPLY_FITNESS_INHERITANCE, BALANCED_TRACES, TRUNCATION_PROPORTION, CROSSOVER_RATE)
    # single_weight_evo_grid_search_experiment(
    #      int(sys.argv[1]), int(sys.argv[2]), remote=False, parallelise=True, hw=False, params=custom_params, gens=1000, static_seed=True, randomise_init_weights=True, sgd=False, dataset_name="ascad"
    # )

    cf = configs[int(sys.argv[1])]
    pool_param = 4 if cf[3] == "cw" else 2

    # neat_experiment(pop_size=250, max_gens=100, remote=True, hw=bool(int(sys.argv[1])), parallelise=True, avg_pooling=bool(int(sys.argv[2])), dataset_name="ascad", only_evolve_hidden=True)
    neat_experiment(pop_size=6, max_gens=cf[0], remote=False, hw=cf[1], parallelise=True, avg_pooling=cf[2], dataset_name=cf[3], only_evolve_hidden=cf[4], noise=cf[5], desync=cf[6], fs_neat=cf[7])
    # neat_experiment(pop_size=6, max_gens=100, remote=False, hw=True, parallelise=True, avg_pooling=True, dataset_name="cw", only_evolve_hidden=False, fs_neat=True)

    # dual_parallel_weight_evo_experiment(sys.argv, remote=False)
    # weight_evo_experiment_from_params(sys.argv, remote=True)

    # attack_chipwhisperer_mlp(remote=False, train_with_ga=True, ass=315, folds=1, shuffle=True, balanced=True,
    #                          psize=1000, gens=100, hw=True, n_dense=1, gen_sgd_train=False, metric=MetricType.CATEGORICAL_CROSS_ENTROPY,
    #                          select_fn="tournament", mut_pow=0.1, mut_rate=0.1)
    # single_ga_experiment(remote_loc=False, use_mlp=True, averaged=False, apply_fi=False, parallelise=True)
    # train_first_layer_ascad_mlp()
    # single_ensemble_experiment()

    # train_and_attack_ascad()
    # nn = build_single_hidden_layer_mlp_ascad(hw=True, avg_pooling=True)
    # nns = [
    #     build_single_hidden_layer_mlp_ascad(hw=False, avg_pooling=False),
    #     build_single_hidden_layer_mlp_ascad(hw=False, avg_pooling=True),
    #     build_single_hidden_layer_mlp_ascad(hw=True, avg_pooling=False),
    #     build_single_hidden_layer_mlp_ascad(hw=True, avg_pooling=True)
    # ]
    # hws = [False, False, True, True]
    # labels = ["ID/NoAP", "ID/AP", "HW/NoAP", "HW/AP"]
    # exp_name = "Single hidden layer with SGD (ASCAD)"
    # train_and_attack_with_multiple_nns(nns, hws, labels, "ascad", exp_name)

    # attack_chipwhisperer_mlp(train_with_ga=False, remote=False, hw=True, n_dense=1, noise=False, desync=0, noise_std=0.05, save=True)

    # exp_names = ["ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.0-xavwi-nosgd", "ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.5-randwi-sgd"]
    # exp_names = ["ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.5-randwi-sgd"]
    # exp_names = ["ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.5_same-folds", "ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_INKE-nofi-balnc-cor0.5_same-folds",
    #              "ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_KEAC-nofi-balnc-cor0.5_same-folds", "ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_KEPR-nofi-balnc-cor0.5_same-folds"]
    # exp_names = ["ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass3584-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.5-randwi-nosgd", "ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass3584-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.5-randwi-sgd"]
    # exp_labels = ["Pure", "Hybrid"]
    # exp_labels = ["ASCAD weight evolution"]
    # file_tag = "ascad_weight_evo"
    # weight_evo_results_from_exp_names(exp_names, exp_labels, file_tag, separate_fit_prog_plots=True)

    # eval_best_nn_from_exp_name("ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass3584-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.5-randwi-nosgd", "ascad", "Final weight evolution NN performance on ASCAD traces")
    # eval_best_nn_from_exp_name("ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.5_same-folds", "cw", "CW pure weight evo.")

    # nn = keras.models.load_model("./trained_models/cw_mlp_trained_sgd.h5")
    # weight_heatmaps_from_exp_name("cw_hybrid_we", exp_name="ps1040-mp0.03-mr0.1-mpdr0.999-fdr0.2-ass315-tsel-tp1.0-mt_CAEN-nofi-balnc-cor0.5-randwi-sgd")

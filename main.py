import multiprocessing as mp

mp.set_start_method("spawn", force=True)
from time import time
import os
import sys

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(2)
# tf.config.threading.set_inter_op_parallelism_threads(1)

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
                         attack_chipwhisperer_mlp, neat_experiment)
from metrics import MetricType
from models import set_nn_load_func
from neat_sca import set_global_data
from result_processing import combine_grid_search_results

set_nn_load_func("mini_mlp_cw")
# set_nn_load_func("mlp_cw", (False, False, 1))
set_global_data(
    "cw", 8000, subkey_idx=1, n_folds=1, remote=False, hw=True,
    metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY, balanced=False,
    use_sgd=True, use_avg_pooling=True
)


def dual_parallel_weight_evo_experiment(args, remote=True):
    pool = mp.Pool(2)

    res = pool.apply_async(single_weight_evo_grid_search_experiment, (int(args[1]), int(args[2]), None, remote))
    res = pool.apply_async(single_weight_evo_grid_search_experiment, (int(args[1]), int(args[2]) + 1, None, remote))

    pool.close()
    pool.join()


if __name__ == "__main__":
    # single_weight_evo_grid_search_experiment(
    #     int(sys.argv[1]), int(sys.argv[2]), remote=False, parallelise=True, hw=True
    # )

    # dual_parallel_weight_evo_experiment(sys.argv, remote=False)
    # weight_evo_experiment_from_params(sys.argv, remote=True)
    # combine_grid_search_results()

    # attack_chipwhisperer_mlp(remote=False, train_with_ga=True, ass=315, folds=1, shuffle=True, balanced=True,
    #                          psize=1000, gens=100, hw=True, n_dense=1, gen_sgd_train=False, metric=MetricType.CATEGORICAL_CROSS_ENTROPY,
    #                          select_fn="tournament", mut_pow=0.1, mut_rate=0.1)
    # single_ga_experiment(remote_loc=False, use_mlp=True, averaged=False, apply_fi=False, parallelise=True)
    # train_first_layer_ascad_mlp()
    # single_ensemble_experiment()

    neat_experiment(pop_size=6, max_gens=3, remote=False, hw=True, parallelise=True, avg_pooling=False, dataset_name="cw")
    # attack_chipwhisperer_mlp(train_with_ga=False, remote=False, hw=True, n_dense=1)

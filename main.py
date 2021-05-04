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
                         attack_chipwhisperer_mlp)
from metrics import MetricType
from models import set_nn_load_func
set_nn_load_func("mlp_cw")
from result_processing import combine_grid_search_results


def dual_parallel_weight_evo_experiment(args, remote=True):
    pool = mp.Pool(2)

    res = pool.apply_async(single_weight_evo_grid_search_experiment, (int(args[1]), int(args[2]), None, remote))
    res = pool.apply_async(single_weight_evo_grid_search_experiment, (int(args[1]), int(args[2]) + 1, None, remote))

    pool.close()
    pool.join()


if __name__ == "__main__":
    # single_weight_evo_grid_search_experiment(
    #     int(sys.argv[1]), int(sys.argv[2]), remote=False, parallelise=True
    # )

    # dual_parallel_weight_evo_experiment(sys.argv, remote=False)
    # weight_evo_experiment_from_params(sys.argv, remote=True)
    # combine_grid_search_results()

    # single_ga_experiment(remote_loc=False, use_mlp=True, averaged=False, apply_fi=False, parallelise=True)
    # train_first_layer_ascad_mlp()
    attack_chipwhisperer_mlp(remote=True, train_with_ga=True, ass=8000, folds=1, shuffle=False, balanced=False, psize=78, gens=150, hw=True, fi=False, metric=MetricType.INCREMENTAL_KEYRANK)
    # attack_chipwhisperer_mlp(remote=False, train_with_ga=False, hw=False)
    # single_ensemble_experiment()

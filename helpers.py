import itertools
import multiprocessing as mp
import os
import pickle

import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy
CCE = CategoricalCrossentropy()

from constants import INVERSE_SBOX, SBOX
from data_processing import shuffle_data
from metrics import keyrank, accuracy, MetricType


def exec_sca(ann_model, x_atk, y_atk, ptexts, true_subkey, subkey_idx=2):
    """
    Executes a side-channel attack on the given traces using the given neural
    network and returns the key rank obtained with the attack.
    """
    # Obtain y_pred_probs for each trace and aggregate them for the key guess
    y_pred_probs = ann_model.predict(x_atk)
    subkey_logprobs = subkey_pred_logprobs(y_pred_probs, ptexts, subkey_idx)

    return keyrank(subkey_logprobs, true_subkey)


def kfold_mean_key_ranks(y_pred_probs, ptexts, true_subkey, k,
                         key_idx=2, experiment_name="", parallelise=False,
                         remote=False):
    """
    Calculates and returns the mean key ranks of the given list of prediction
    probability arrays, the data set's metadata, and the desired number of
    folds.
    """
    # For each fold, store the key rank for all attack set sizes
    set_size = len(y_pred_probs)
    fold_key_ranks = np.zeros((set_size, k), dtype=np.uint8)

    # For both the parallel and sequential methods, the core idea is to reuse
    # subsets of the predictions to simulate attacks over different folds
    if parallelise:
        pool = mp.Pool(get_pool_size(remote))

        # Compute key ranks for each trace amount in parallel
        shuffled = [shuffle_data(y_pred_probs, ptexts) for i in range(k)]
        argss = [
            (i, shuffled[i][0], shuffled[i][1], key_idx, set_size, true_subkey)
            for i in range(k)
        ]
        # Result = 2D array indexed with [fold, trace_idx]
        map_results = pool.starmap(compute_fold_keyranks, argss)
        for fold in range(k):
            fold_key_ranks[:, fold] = map_results[fold]
    else:
        for fold in range(k):
            y_pred_probs, ptexts = shuffle_data(y_pred_probs, ptexts)
            fold_key_ranks[:, fold] = compute_fold_keyranks(
                fold, y_pred_probs, ptexts, key_idx, set_size, true_subkey
            )

    if parallelise:
        pool.close()
        pool.join()

    # Build an array that contains the mean key rank for each trace amount
    mean_key_ranks = np.zeros(set_size, dtype=np.uint8)
    for i in range(set_size):
        mean_key_ranks[i] = round(np.mean(fold_key_ranks[i]))
    filepath = f"results/{experiment_name}_test_set_mean_key_ranks.pickle"
    with open(filepath, "wb") as f:
        pickle.dump(mean_key_ranks, f)

    # Print key ranks for various attack set sizes
    atk_set_sizes = range(set_size + 1, 100)
    n_atk_set_sizes = len(atk_set_sizes)
    reached_keyrank_zero = False
    for (n_traces, rank) in enumerate(mean_key_ranks):
        # Print milestones
        if n_traces % 250 == 0:
            print(f"Mean key rank with {n_traces + 1} attack traces: {rank}")
        if rank == 0 and not reached_keyrank_zero:
            print(f"Key rank 0 obtained after {n_traces + 1} traces")
            reached_keyrank_zero = True
    
    return mean_key_ranks


def compute_fold_keyranks(fold, y_pred_probs, atk_ptexts, subkey_idx,
                          atk_set_size, true_subkey, verbose=True):
    """
    Computes the key ranks for each amount of traces for a given fold of attack
    traces, which are assumed to already be shuffled if necessary.
    """
    if verbose:
        print(f"Obtaining key ranks for fold {fold}...")
    fold_key_ranks = np.zeros(atk_set_size, dtype=np.uint8)

    # Track the summed log probability of each subkey candidate
    subkey_logprobs = np.zeros(256)

    # Iterate over each list of 256 probabilities in y_pred_probs.
    # Each list corresponds to the predictions of 1 trace.
    for (i, pred_probs) in enumerate(y_pred_probs):
        pt = atk_ptexts[i][subkey_idx]

        # Convert each label to a subkey and add its logprob to the sum
        for (label, label_pred_prob) in enumerate(pred_probs):
            subkey = label_to_subkey(pt, label)

            # Avoid computing np.log(0), which returns -inf
            logprob = np.log(label_pred_prob) if label_pred_prob > 0 else 0
            subkey_logprobs[subkey] += logprob
    
        # Note that index i stores the key rank obtained after (i+1) traces
        fold_key_ranks[i] = keyrank(subkey_logprobs, true_subkey)

    return fold_key_ranks


def compute_fitness(nn, x_atk, y_atk, ptexts, metric_type, true_subkey,
                    atk_set_size, subkey_idx=2):
    """
    Executes a side-channel attack on the given traces using the given neural
    network and uses the obtained prediction probabilities to compute the key
    rank and/or accuracy.
    """
    y_pred_probs = nn.predict(x_atk)
    return evaluate_preds(
        y_pred_probs, metric_type, ptexts, true_subkey, y_atk, atk_set_size,
        subkey_idx
    )


def evaluate_preds(preds, metric_type, ptexts, true_subkey, true_labels,
                   set_size, subkey_idx=2):
    """
    Evaluates the given predictions using the method indicated by the given
    metric type and returns the result.
    """
    if metric_type == MetricType.KEYRANK:
        subkey_probs = subkey_pred_logprobs(preds, ptexts, subkey_idx)
        return keyrank(subkey_probs, true_subkey)
    elif metric_type == MetricType.ACCURACY:
        return (1 - accuracy(preds, true_labels))*100
    elif metric_type == MetricType.KEYRANK_AND_ACCURACY:
        subkey_probs = subkey_pred_logprobs(preds, ptexts, subkey_idx)
        res = keyrank(subkey_probs, true_subkey) - accuracy(preds, true_labels)
        return res
    elif metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        return CCE(true_labels, preds).numpy()  # true_labels should be 1-hot
    elif metric_type == MetricType.INCREMENTAL_KEYRANK:
        # f = (n_traces_for_kr_zero + kr_10_pct + 0.5*kr_50_pct)
        key_ranks = compute_fold_keyranks(
            7, preds, ptexts, subkey_idx, set_size, true_subkey, verbose=False)

        kr0_n_traces = first_zero_value_idx(key_ranks, set_size)/(set_size - 1)
        kr_10pct = min(key_ranks[round(set_size*0.1) - 1], 128)/128
        kr_50pct = min(key_ranks[round(set_size*0.5) - 1], 128)/128
        acc = accuracy(preds, true_labels)

        return kr0_n_traces + kr_10pct + 0.5*kr_50pct + 0.5*(1 - acc)
    elif metric_type == MetricType.KEYRANK_PROGRESS:
        key_ranks = compute_fold_keyranks(
            7, preds, ptexts, subkey_idx, set_size, true_subkey, verbose=False)
        return np.mean(np.diff(key_ranks.astype(np.int16)))
    else:
        print("Encountered invalid metric type. Quitting.")
        exit(1)


def compute_label(ptext, subkey, mask=0):
    """
    Computes and returns the label, which is equivalent to the SBOX output, for
    a trace with a given plaintext and the targeted subkey.
    """
    return SBOX[ptext ^ subkey ^ mask]


def label_to_subkey(ptext, label, mask=0):
    """
    Retrieves the predicted subkey based on a predicted label and the known
    plaintext corresponding to some trace.
    """
    return INVERSE_SBOX[label] ^ ptext ^ mask


def subkey_pred_logprobs(label_pred_probs, ptexts, subkey_idx=2, masks=None):
    """
    Computes the logarithm of the prediction probability of each subkey
    candidate. This is done by iterating over each trace's label prediction
    probabilities and retrieving the corresponding subkey prediction
    probabilities.
    """
    # Track the summed log probability of each subkey candidate
    subkey_logprobs = np.zeros(256)

    # Iterate over each list of 256 probabilities in label_pred_probs
    for (i, pred_probs) in enumerate(label_pred_probs):
        pt = ptexts[i][subkey_idx]

        # Convert each label to a subkey and add its logprob to the sum
        for (label, label_pred_prob) in enumerate(pred_probs):
            subkey = label_to_subkey(pt, label)

            # Avoid computing np.log(0), which returns -inf
            # Note: ASCAD devs solve this by defaulting to min(pred_probs)**2
            logprob = np.log(label_pred_prob) if label_pred_prob > 0 else 0  # TODO: Change to ASCAD implementation
            subkey_logprobs[subkey] += logprob
    
    return subkey_logprobs


def compute_mem_req(pop_size, nn, atk_set_size):
    """
    Approximates the amount of memory that running a GA instance will require
    based on the population size and the amount of network weights.

    The computation assumes that:
        - Each weight requires 4 bytes;
        - Fitness evaluation spawns a new process for each individual,
        copying its weights to construct a model of roughly equal size.
        This results in (n_weights * 3) weights per individual;
        - Fitness evaluation copies the required items from the attack set;
        - Each item in the attack set is an array of 700 64-bit floats.
    """
    max_indivs = pop_size*2
    ws = nn.get_weights()
    n_ws = np.sum([layer_ws.size for layer_ws in ws])

    ws_bytes = 4*n_ws
    atk_set_bytes = atk_set_size*700  # Times 8 if scaling traces to 64b floats

    return max_indivs*(ws_bytes*3 + atk_set_bytes)


def compute_mem_req_from_known_vals(pop_size, data_set_size, scaling=True):
    """
    Approximates the amount of memory that running a GA instance will require
    based on the population size and the amount of network weights. This is
    done by using 0.4GB and 0.000003 as constants for the required size per
    individual and per data set size unit respectively.

    The estimated 0.4GB per individual includes the amount of memory used to
    spawn a new process.

    Returns:
        The approximate RAM requirement in GB.
    """
    return 2*pop_size*(0.4 + data_set_size*0.000003*(8 if scaling else 1))


def compute_mem_req_for_pop(pop_size, data_set_size, n_trace_points,
                            n_indiv_weights, scaling=True):
    trace_size = n_trace_points*(8 if scaling else 1)
    label_size = 1  # Assumes full list of labels is only for 1 subkey
    pt_size = 16  # 16 bytes per full plaintext
    k_size = 8  # Only pass the required key byte for a single subkey index
    indiv_size = 4*n_indiv_weights

    data_size = data_set_size*(trace_size + label_size + pt_size) + k_size
    return (2*pop_size*(data_size + indiv_size))/1e6  # Return in MB


def load_model_weights_from_ga_results(experiment_name):
    """
    Loads and returns the weights of the best individual constructed during the
    GA experiment with the given experiment name.
    """
    path = f"results/{experiment_name}_ga_results.pickle"
    nn_weights = None
    with open(path, "rb") as f:
        ga_results = pickle.load(f)
        nn_weights = ga_results[0].weights
    
    return nn_weights


def gen_experiment_name(pop_size, atk_set_size, select_fun):
    """
    Generates an experiment name for a GA run using the given parameters.
    """
    return f"ps{pop_size}-ass{atk_set_size}-{select_fun[0]}select"


def gen_extended_exp_name(ps, mp, mr, mpdr, fdr, ass, sf, mt, nn):
    """
    Generates an experiment name for a GA run using the given arguments.

    Arguments:
        ps: Population size.
        mp: Mutation power.
        mr: Mutation rate.
        mpdr: Mutation power decay rate.
        fdr: Fitness inheritance decay rate.
        ass: Attack set size.
        sf: Selection function.
        mt: Metric type.
        nn: Neural network model name.
    """
    return f"ps{ps}-mp{mp}-mr{mr}-mpdr{mpdr}-fdr{fdr}-ass{ass}-sf_{sf[0]}-mt_{mt.id()}-{nn}"


def calc_max_fitness(metric_type):
    """
    Returns the maximum fitness based on the given metric type.
    """
    return 100 if metric_type == MetricType.ACCURACY else 255
    mapping = {
        MetricType.KEYRANK: 255,
        MetricType.KEYRANK_AND_ACCURACY: 255,
        MetricType.ACCURACY: 100,
        MetricType.INCREMENTAL_KEYRANK: 3,
        MetricType.KEYRANK_PROGRESS: np.inf,
    }
    return mapping[metric_type]


def calc_min_fitness(metric_type):
    """
    Returns the maximum fitness based on the given metric type.
    """
    mapping = {
        MetricType.KEYRANK: 0,
        MetricType.KEYRANK_AND_ACCURACY: -1,
        MetricType.ACCURACY: 0,
        MetricType.INCREMENTAL_KEYRANK: 0,
        MetricType.KEYRANK_PROGRESS: -2.5,
    }
    return mapping[metric_type]


def get_pool_size(remote, pop_size=50):
    """
    Determines and returns the size of the multiprocessing pool based on
    whether the application is running locally or remotely.
    """
    return min(pop_size*2, len(os.sched_getaffinity(0))) if remote \
        else round(min(pop_size*2, mp.cpu_count()*0.75))


def gen_ga_grid_search_arg_lists():
    """
    Returns a list of lists, where each inner list contains arguments for a
    single grid search run for the weight evolution GA experiments.

    These argument combinations result in 486 configurations, each of which
    would likely result in a runtime below 4 hours on the TUD HPC cluster.
    """
    # Total amount of experiments: 486
    pop_sizes = [25, 50, 75]  # 3 values
    mut_pows = [0.01, 0.03, 0.05]  # 3 values
    mut_rates = [0.01, 0.04, 0.07]  # 3 values
    mut_pow_dec_rates = [0.99, 0.999, 1.0]  # 3 values
    fi_dec_rates = [0.0]  # 1 value
    atk_set_sizes = [16, 64, 256]  # 3 values
    selection_methods = ["tournament", "roulette_wheel"]  # 2 values
    metrics = [MetricType.KEYRANK_AND_ACCURACY]  # 1 value

    argss = [
        tup for tup in itertools.product(
            pop_sizes, mut_pows, mut_rates, mut_pow_dec_rates,
            fi_dec_rates, atk_set_sizes, selection_methods, metrics
        )
    ]

    return argss


def gen_max_resources_ga_grid_search_arg_lists():
    """
    Returns a list of lists, where each inner list contains arguments for a
    single grid search run for the weight evolution GA experiments.

    These argument combinations result in 216 configurations, many of which
    would likely result in a runtime above 4 hours on the TUD HPC cluster.
    """
    # Total amount of experiments: 216
    pop_sizes = [250]  # 1 value
    mut_pows = [0.03, 0.06, 0.09]  # 3 values
    mut_rates = [0.04, 0.07, 0.10]  # 3 values
    mut_pow_dec_rates = [0.99, 0.999]  # 2 values
    fi_dec_rates = [0.2]  # 1 value # TODO: Implement FI correctly for non-cloned individuals, i.e. give them a parent fitness based on previous gen's performance
    atk_set_sizes = [5120]  # 1 value
    selection_methods = ["tournament", "roulette_wheel", "unbiased_tournament"]  # 3 values
    metrics = [MetricType.INCREMENTAL_KEYRANK]  # 1 value
    apply_fi_modes = [False, True]  # 2 values
    balanced_traces = [True, False]  # 2 values

    argss = [
        tup for tup in itertools.product(
            pop_sizes, mut_pows, mut_rates, mut_pow_dec_rates,
            fi_dec_rates, atk_set_sizes, selection_methods, metrics,
            apply_fi_modes, balanced_traces
        )
    ]

    return argss


def first_idx_with_value(a, x, a_len=None):
    """
    Finds the index in array a where element x first appears. Returns the last
    index if the element is not found.
    """
    for (i, n) in enumerate(a):
        if n == x:
            return i
    return a_len - 1 or len(a) - 1


def first_zero_value_idx(a, a_len=None):
    """
    finds the index in array a where value 0 first appears.
    """
    return first_idx_with_value(a, 0, a_len)

import pickle

import numpy as np

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


def bag_ensemble_predictions(nns, x_atk, y_atk, ptexts, true):
    """
    Executes a side-channel attack on the given traces by summing the
    predictions from the given neural networks.

    Returns the summed predictions 
    """


def kfold_mean_key_ranks(y_pred_probs, atk_ptexts, true_subkey, k,
                         subkey_idx=2, experiment_name=""):
    """
    Calculates and returns the mean key ranks of the given list of prediction
    probability arrays, the data set's metadata, and the desired number of
    folds.
    """
    # For each fold, store the key rank for all attack set sizes
    atk_set_size = len(y_pred_probs)
    fold_key_ranks = np.zeros((atk_set_size, k), dtype=np.uint8)

    # Reuse subsets of the predictions to simulate attacks over different folds
    for fold in range(k):
        print(f"Obtaining key ranks for fold {fold}...")
        y_pred_probs, atk_ptexts = shuffle_data(y_pred_probs, atk_ptexts)

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
            fold_key_ranks[i, fold] = keyrank(subkey_logprobs, true_subkey)

    # Build an array that contains the mean key rank for each trace amount
    mean_key_ranks = np.zeros(atk_set_size, dtype=np.uint8)
    for i in range(atk_set_size):
        mean_key_ranks[i] = round(np.mean(fold_key_ranks[i]))
    with open(f"{experiment_name}_test_set_mean_key_ranks.pickle", "wb") as f:
        pickle.dump(mean_key_ranks, f)

    # Print key ranks for various attack set sizes
    atk_set_sizes = range(atk_set_size + 1, 100)
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


def compute_fitness(nn, x_atk, y_atk, ptexts, metric_type, true_subkey,
                    subkey_idx=2):
    """
    Executes a side-channel attack on the given traces using the given neural
    network and uses the obtained prediction probabilities to compute the key
    rank and/or accuracy.

    Returns fitness as keyrank, (1 - accuracy)*100, or (keyrank - accuracy)
    depending on the given metric type.
    """
    y_pred_probs = nn.predict(x_atk)
    return evaluate_preds(
        y_pred_probs, metric_type, ptexts, true_subkey, y_atk, subkey_idx
    )


def evaluate_preds(preds, metric_type, ptexts, true_subkey, true_labels,
                   subkey_idx=2):
    """
    Evaluates the given predictions using the method indicated by the given
    metric type and returns the result.

    Arguments:
        preds: A 2-dimensional array, where the i-th array is a list of
        prediction probabilities for the i-th trace.
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
            logprob = np.log(label_pred_prob) if label_pred_prob > 0 else 0
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


def compute_mem_req_from_known_vals(pop_size, data_set_size, scaling=False):
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


def load_model_weights_from_ga_results(experiment_name):
    """
    Loads and returns the weights of the best individual constructed during the
    GA experiment with the given experiment name.
    """
    path = f"{experiment_name}_ga_results.pickle"
    nn_weights = None
    with open(path, "rb") as f:
        ga_results = pickle.load(f)
        nn_weights = ga_results[0]
    
    return nn_weights


def gen_experiment_name(pop_size, atk_set_size, select_fun):
    """
    Generates an experiment name for a GA run using the given parameters.
    """
    return f"ps{pop_size}-ass{atk_set_size}-{select_fun[0]}select"


def calc_max_fitness(metric_type, apply_fi=False, fi_decay=0.2):
    """
    Returns the maximum fitness based on the given metric type.
    """
    return 100 if metric_type == MetricType.ACCURACY else 255


import pickle

import numpy as np

from constants import INVERSE_SBOX, SBOX
from metrics import keyrank


def exec_sca(ann_model, x_atk, y_atk, ptexts, true_subkey, subkey_idx=2):
    """
    Executes a side-channel attack on the given traces using the given neural
    network and returns the key rank obtained with the attack.
    """
    # Obtain y_pred_probs for each trace and aggregate them for the key guess
    y_pred_probs = ann_model.predict(x_atk)
    subkey_logprobs = subkey_pred_logprobs(y_pred_probs, ptexts, subkey_idx)

    return keyrank(subkey_logprobs, true_subkey)


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


def gen_experiment_name(pop_size, atk_set_size):
    """
    Generates an experiment name for a GA run using the given parameters.
    """
    return f"./results/ps{pop_size}-ass{atk_set_size}"

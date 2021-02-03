import numpy as np

from constants import INVERSE_SBOX, SBOX
from metrics import keyrank


def exec_sca(ann_model, x_atk, y_atk, ptexts, true_subkey, subkey_idx=2):
    """
    Executes a side-channel attack on the given traces using the given neural
    network and returns the key rank obtained with the attack.
    """
    # Obtain y_pred_probs for each trace and aggregate them for the key guess
    y_pred_probs = ann_model.predict_proba(x_atk)
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

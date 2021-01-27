import numpy as np


def keyrank(subkey_probs, true_subkey):
    """
    Sorts the given subkey probabilities descendingly and returns the rank
    (index) of the actual subkey.
    """
    # Obtain a descendingly sorted list of subkey candidates
    sorted_subkeys = np.argsort(subkey_probs)[::-1]
    return list(sorted_subkeys).index(true_subkey)

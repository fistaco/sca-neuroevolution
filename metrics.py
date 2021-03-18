from enum import Enum

import numpy as np


def keyrank(subkey_probs, true_subkey):
    """
    Sorts the given subkey probabilities descendingly and returns the rank
    (index) of the actual subkey.
    """
    # Obtain a descendingly sorted list of subkey candidates
    sorted_subkeys = np.argsort(subkey_probs)[::-1]
    return list(sorted_subkeys).index(true_subkey)


def accuracy(label_pred_probs, true_labels):
    """
    Computes the prediction accuracy for some trace set by comparing the given
    arrays of label prediction probabilities with the list of true labels.
    """
    n_preds = len(true_labels)

    acc = 0
    for i in range(n_preds):
        label_pred = np.argmax(label_pred_probs[i])
        if label_pred == true_labels[i]:
            acc += 1/n_preds

    return np.float64(acc)


class MetricType(Enum):
    KEYRANK = 0
    ACCURACY = 1
    KEYRANK_AND_ACCURACY = 2

    def id(self):
        substrs = self.name.split("_")
        if len(substrs) == 1:
            return self.name[:3]
        else:
            return f"{substrs[0][:2] + substrs[-1][:2]}"
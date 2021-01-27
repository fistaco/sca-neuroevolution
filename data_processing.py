import h5py
import numpy as np


def load_ascad_data(data_filepath="./../ASCAD_data/ASCAD_databases/ASCAD.h5",
                    load_metadata=False):
    """
    Loads the ASCAD data set with h5py and returns a tuple containing the
    training traces, training labels, attack traces, attack labels, and
    optionally the metadata.
    """
    input_file = h5py.File(data_filepath, "r")

    # Load traces and their corresponding labels
    # Convert them to numpy arrays for compatibility with keras
    train_traces = np.array(input_file['Profiling_traces/traces'], dtype=np.int8)
    train_labels = np.array(input_file['Profiling_traces/labels'])
    atk_traces = np.array(input_file['Attack_traces/traces'], dtype=np.int8)
    atk_labels = np.array(input_file['Attack_traces/labels'])

    data_tup = (train_traces, train_labels, atk_traces, atk_labels)

    if load_metadata:
        # Contains ptexts, keys, ctexts, masks, and desync bool for each trace.
        train_metadata = input_file['Profiling_traces/metadata']
        atk_metadata = input_file['Attack_traces/metadata']

        data_tup = data_tup + (train_metadata, atk_metadata)

    # input_file.close()  # hdf5 data usage requires the file to remain open

    return data_tup


def reshape_input_for_cnns(input_data):
    """
    Converts and returns the given 2D input data so that it consists of
    singleton arrays to make it eligible for CNNs.
    """
    return input_data.reshape((input_data.shape[0], input_data.shape[1], 1))


def sample_data(n_samples, set_x, set_y):
    """
    Extracts and returns n random samples from the 2 given data sets. 
    """
    indices = np.random.choice(len(set_x), n_samples)

    return (set_x[indices], set_y[indices])


def train_test_split(x, y, train_proportion):
    """
    Splits data sets x and y into train and test sets according to the given
    proportion.
    """
    cutoff = int(len(x)*train_proportion)
    return (x[:cutoff], x[cutoff:], y[:cutoff], y[cutoff:])

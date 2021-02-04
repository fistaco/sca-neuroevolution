import h5py
import numpy as np

from sklearn import preprocessing


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


def load_ascad_atk_variables(subkey_idx=2, for_cnns=True, scale=True):
    """
    Loads the ASCAD data set and returns all variables required to perform an
    SCA with a neural network. If the attack will be carried out by a CNN,
    the attack set is reshaped first.

    Returns:
        A tuple containing attack traces, attack labels, the true target key at
        the given subkey index, and the attack plaintexts.
    """
    (x, y, x_atk, y_atk, train_meta, atk_meta) = \
        load_ascad_data(load_metadata=True)
    original_input_shape = (700, 1)

    # Declare easily accessible variables for relevant metadata
    target_atk_subkey = atk_meta['key'][0][subkey_idx]
    atk_ptexts = atk_meta['plaintext']

    # Scale the inputs to range [0, 1] if desired
    if scale:
        x_atk = scale_inputs(x_atk)

    # Reshape trace inputs for CNN compatibility if necessary
    if for_cnns:
        x_atk = x_atk.reshape((x_atk.shape[0], x_atk.shape[1], 1))

    return (x_atk, y_atk, target_atk_subkey, atk_ptexts)


def reshape_input_for_cnns(input_data):
    """
    Converts and returns the given 2D input data so that it consists of
    singleton arrays to make it eligible for CNNs.
    """
    return input_data.reshape((input_data.shape[0], input_data.shape[1], 1))


def sample_data(n_samples, *data_sets):
    """
    Extracts and returns n random samples from the given data sets. The same
    random indices are used to extract samples from each set.
    """
    indices = np.random.choice(len(data_sets[0]), n_samples)

    tup = ()
    for data_set in data_sets:
        tup += (data_set[indices], )

    return tup


def shuffle_data(*data_sets):
    """
    Shuffles each given data set according the same randomly generated indices
    and returns a tuple containing the shuffled data sets.
    """
    perm = np.random.permutation(len(data_sets[0]))

    tup = ()
    for data_set in data_sets:
        tup += (data_set[perm], )
    
    return tup


def train_test_split(x, y, train_proportion):
    """
    Splits data sets x and y into train and test sets according to the given
    proportion.
    """
    cutoff = int(len(x)*train_proportion)
    return (x[:cutoff], x[cutoff:], y[:cutoff], y[cutoff:])


def scale_inputs(inputs):
    """
    Scales the given inputs uniformly to put them in range [0, 1].

    Arguments:
        xs: A 2-dimensional array of trace inputs.
    """
    return preprocessing.MinMaxScaler((0, 1)).fit_transform(inputs)

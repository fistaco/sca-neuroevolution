import h5py
import numpy as np

from keras.utils import to_categorical
from sklearn import preprocessing


def load_ascad_data(data_filepath="./../ASCAD_data/ASCAD_databases/ASCAD.h5",
                    load_metadata=False, remote_loc=False):
    """
    Loads the ASCAD data set with h5py and returns a tuple containing the
    training traces, training labels, attack traces, attack labels, and
    optionally the metadata.
    """
    # Use the absolute directory from the TUD HPC project dir if desired
    if remote_loc:
        data_filepath = "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/" + \
                        "fschijlen/ASCAD_data/ASCAD_databases/ASCAD.h5"

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


def load_ascad_atk_variables(subkey_idx=2, for_cnns=True, scale=False):
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


def load_prepared_ascad_vars(subkey_idx=2, scale=True, use_mlp=False,
                             remote_loc=False, for_sgd=False):
    """
    Loads the ASCAD training and attack traces along with the metadata, applies
    reshaping for CNNs, scales the traces, and returns all relevant variables
    for instant use.

    Note that y_train is not converted for training with SGD.
    """
    (x_train, y_train, x_atk, y_atk, train_meta, atk_meta) = \
        load_ascad_data(load_metadata=True, remote_loc=remote_loc)
    original_input_shape = (700, 1)
    x_train, y_train = x_train[:45000], y_train[:45000]

    # Declare easily accessible variables for relevant metadata
    target_train_subkey = train_meta['key'][0][subkey_idx]
    train_ptexts = train_meta['plaintext']
    target_atk_subkey = atk_meta['key'][0][subkey_idx]
    atk_ptexts = atk_meta['plaintext']

    # Convert labels to one-hot encoding probabilities if training with SGD
    if for_sgd:
        y_train = to_categorical(y_train, num_classes=256)

    # Scale all trace inputs to [low, 1]
    low_bound = -1 if use_mlp else 0
    x_train = scale_inputs(x_train, low_bound)
    x_atk = scale_inputs(x_atk, low_bound)

    # Reshape the trace input to come in singleton arrays
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_atk = x_atk.reshape((x_atk.shape[0], x_atk.shape[1], 1))

    return (x_train, y_train, train_ptexts, target_train_subkey, x_atk, \
        y_atk, atk_ptexts, target_atk_subkey)


def load_dpav4(subkey_idx=0):
    pass


def load_chipwhisperer_data(n_train=8000, subkey_idx=0):
    """
    Loads the Chipwhisperer data set and returns it as a tuple containing
    (x_train, y_train, pt_train, x_atk, y_atk, pt_atk, k).
    """
    dir_path = "./../Chipwhisperer/"
    x = np.load(f"{dir_path}traces.npy")[:10000]
    y = np.load(f"{dir_path}labels.npy")[:10000]
    pt = np.load(f"{dir_path}plain.npy")[:10000]
    k = np.load(f"{dir_path}key.npy")[0][subkey_idx]

    x = reshape_nn_input(x)

    n = n_train  # End index of train traces and start index of attack traces

    return (x[:n], y[:n], pt[:n], x[n:], y[n:], pt[n:], k)


def reshape_nn_input(input_data):
    """
    Converts and returns the given 2D input data so that it consists of
    singleton arrays to make it compatible with various NNs.
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


def balanced_sample(n_samples, x, y, z, n_classes=256, shuffle=False):
    """
    Obtains a balanced sample of a given size from sets x, y, and z according
    to the first indices with occurrences of unique values in set y.
    """
    assert n_samples % n_classes == 0, "#samples is not a multiple of #classes"

    if shuffle:
        x, y, z = shuffle_data(x, y, z)

    idxs = np.zeros(n_samples, dtype=np.int32)
    n_samples_per_class = n_samples//n_classes
    uniq, uidxs, counts = np.unique(y, return_index=True, return_counts=True)

    assert n_samples_per_class < counts.min(), "Too few unique samples"

    # Repeatedly add, mask, and rediscover known indices of unique values
    y_m = np.ma.array(y, mask=False)
    for i in range(n_samples_per_class):
        start, end = i*n_classes, (i + 1)*n_classes
        idxs[start:end] = uidxs
        y_m[uidxs] = np.ma.masked

        uidxs = np.unique(y_m, return_index=True)[1][:-1]  # Ignore mask at -1

    return (x[idxs], y[idxs], z[idxs])


def train_test_split(x, y, train_proportion):
    """
    Splits data sets x and y into train and test sets according to the given
    proportion.
    """
    cutoff = int(len(x)*train_proportion)
    return (x[:cutoff], x[cutoff:], y[:cutoff], y[cutoff:])


def scale_inputs(inputs, low=0):
    """
    Scales the given inputs uniformly to put them in range [low, 1].

    Arguments:
        xs: A 2-dimensional array of trace inputs.
    """
    return preprocessing.MinMaxScaler((low, 1)).fit_transform(inputs)


def to_uint8(data_set):
    """
    Converts the given data set's values to fit within the range
    [0, 1, ..., 255], sets the datatype as uint8 and returns the result.

    This method assumes the original data values are 8-bit integers ranging
    from -128 to 127.
    """
    return (data_set + 128).astype(np.uint8)

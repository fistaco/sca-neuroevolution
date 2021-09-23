import h5py
import numpy as np

from keras.utils import to_categorical
from sklearn import preprocessing

from constants import HW


def load_data(dataset_name, hw=False, remote=False, noise_std=0.0, desync=0):
    load_funcs = {
        "ascad": load_prepared_ascad_vars,
        "ascad_desync50": 7,
        "ascad_desync100": 7,
        "cw": load_chipwhisperer_data,
        "dpav4": load_dpav4
    }

    # Load (x_train, y_train, pt_train, k_train, x_atk, y_atk, pt_atk, k_atk).
    # Some methods only return 1 key, in which case we reformat the tuple.
    x = list(load_funcs[dataset_name](hw=hw, remote=remote))

    # Load train and attack traces with countermeasures if desired
    if noise_std > 0.0:
        # x[0] = np.load(f"{dataset_name}_train_traces_noisy.npy")
        # x[3] = np.load(f"{dataset_name}_atk_traces_noisy.npy")
        x[0] = apply_noise(x[0], std=noise_std)
        x[3] = apply_noise(x[3], std=noise_std)

    if desync > 0:
        # x[0] = np.load(f"{dataset_name}_train_traces_desync{desync}.npy")
        # x[3] = np.load(f"{dataset_name}_atk_traces_desync{desync}.npy")
        x[0] = apply_desync(x[0], desync_level=desync)
        x[3] = apply_desync(x[3], desync_level=desync)

    if len(x) == 7:
        return (x[0], x[1], x[2], x[-1], x[3], x[4], x[5], x[6])

    return tuple(x)


def commonly_used_subkey_idx(dataset_name):
    """
    Returns the most commonly used subkey index for the data set with the given
    name.
    """
    mapping = {
        "ascad": 2,
        "cw": 1
    }
    return mapping[dataset_name]


def load_ascad_data(data_filepath="./../ASCAD_data/ASCAD_databases/ASCAD.h5",
                    load_metadata=False, remote_loc=False, desync=0):
    """
    Loads the ASCAD data set with h5py and returns a tuple containing the
    training traces, training labels, attack traces, attack labels, and
    optionally the metadata.
    """
    set_name = "ASCAD"
    if desync > 0:
        set_name = f"ASCAD_desync{desync}"

    # Use the absolute directory from the TUD HPC project dir if desired
    if remote_loc:
        data_filepath = f"/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/" + \
                        f"fschijlen/ASCAD_data/ASCAD_databases/{set_name}.h5"
    else:
        data_filepath = f"./../ASCAD_data/ASCAD_databases/{set_name}.h5"

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


def load_prepared_ascad_vars(subkey_idx=2, scale=True, use_mlp=True,
                             remote=False, for_sgd=False, hw=False,
                             n_train=45000, desync=0):
    """
    Loads the ASCAD training and attack traces along with the metadata, applies
    reshaping for CNNs, scales the traces, and returns all relevant variables
    for instant use.

    Note that y_train is not converted for training with SGD.
    """
    (x_train, y_train, x_atk, y_atk, train_meta, atk_meta) = \
        load_ascad_data(load_metadata=True, remote_loc=remote, desync=desync)
    original_input_shape = (700, 1)
    x_train, y_train = x_train[:n_train], y_train[:n_train]

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

    if hw:
        y_train = to_hw(y_train)
        y_atk = to_hw(y_atk)

    return (x_train, y_train, train_ptexts, target_train_subkey, x_atk, \
        y_atk, atk_ptexts, target_atk_subkey)


def load_dpav4(subkey_idx=0, hw=False, remote=False):
    dir_path = "./../dpav4/" if not remote \
        else "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/fschijlen/dpav4/"

    x_train = np.load(f"{dir_path}profiling_traces_dpav4.npy")
    y_train = np.squeeze(
        np.load(f"{dir_path}profiling_labels_dpav4.npy").astype(np.uint8))
    pt_train = np.load(f"{dir_path}profiling_plaintext_dpav4.npy") \
        .astype(np.uint8)[:, subkey_idx]
    x_atk = np.load(f"{dir_path}attack_traces_dpav4.npy")
    y_atk = np.squeeze(
        np.load(f"{dir_path}attack_labels_dpav4.npy").astype(np.uint8))
    pt_atk = np.load(f"{dir_path}attack_plaintext_dpav4.npy") \
        .astype(np.uint8)[:, subkey_idx]
    k = np.uint8(np.load(f"{dir_path}key.npy")[subkey_idx])
    m = np.uint8(np.load(f"{dir_path}mask.npy")[subkey_idx])

    if hw:
        y_train = to_hw(y_train)
        y_atk = to_hw(y_atk)

    return (x_train, y_train, pt_train, x_atk, y_atk, pt_atk, k)


def load_chipwhisperer_data(n_train=8000, subkey_idx=1, remote=False,
                            hw=False):
    """
    Loads the Chipwhisperer data set and returns it as a tuple containing
    (x_train, y_train, pt_train, x_atk, y_atk, pt_atk, k).
    """
    dir_path = "./../Chipwhisperer/" if not remote else \
        "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/fschijlen/Chipwhisperer/"
    x = np.load(f"{dir_path}traces.npy")[:10000]
    y = np.load(f"{dir_path}labels.npy")[:10000]
    pt = np.load(f"{dir_path}plain.npy")[:10000]
    k = np.load(f"{dir_path}key.npy")[0][subkey_idx]

    x = reshape_nn_input(x)

    if hw:
        for i in range(10000):
            y[i] = HW[y[i]]

    n = n_train  # End index of train traces and start index of attack traces

    return (x[:n], y[:n], pt[:n], x[n:], y[n:], pt[n:], k)


def reshape_nn_input(input_data):
    """
    Converts and returns the given 2D input data so that it consists of
    singleton arrays to make it compatible with various NNs.
    """
    return input_data.reshape((input_data.shape[0], input_data.shape[1], 1))


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


def sample_traces(n_samples, x, y, z, n_classes=256, shuffle=True,
                  balanced=False):
    """
    Samples `n_samples` of traces from the given sets of traces, labels, and
    plaintexts, optionally balancing them by label value.
    """
    if balanced:
        return balanced_sample(n_samples, x, y, z, n_classes, shuffle)
    else:
        return sample_data(n_samples, x, y, z)


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


def balanced_sample(n_samples, x, y, z, n_classes=256, shuffle=True):
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

    assert n_samples_per_class <= counts.min(), \
        f"Too few unique samples ({counts.min()})"

    # Repeatedly add, mask, and rediscover known indices of unique values
    y_m = np.ma.array(y, mask=False)
    for i in range(n_samples_per_class):
        start, end = i*n_classes, (i + 1)*n_classes
        idxs[start:end] = uidxs
        y_m[uidxs] = np.ma.masked

        uidxs = np.unique(y_m, return_index=True)[1][:-1]  # Ignore mask at -1

    if shuffle:
        return shuffle_data(x[idxs], y[idxs], z[idxs]) 

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


def to_hw(labels):
    """
    Converts all `labels` to their Hamming weight and returns the set.
    """
    for i in range(len(labels)):
        labels[i] = HW[labels[i]]

    return labels


def apply_noise(traces, mean=0.0, std=0.05):
    """
    Applies Gaussian noise to the given `traces` according to the given `mean`
    and `variance`. It is assumed that `traces` is a multidimensional numpy
    array of values in the range [-1, 1].

    This method implements the Gaussian noise algorithm from "Remove Some
    Noise: On Pre-processing of Side-channel Measurements with Autoencoders" by
    Wu et al.
    """
    return traces + np.random.normal(mean, std, traces.shape)


def apply_desync(traces, desync_level=50):
    """
    Desynchronises the given `traces` by displacing each trace point in each
    trace by up to `desync_level` indices.

    This method implements the Gaussian noise algorithm from "Remove Some
    Noise: On Pre-processing of Side-channel Measurements with Autoencoders" by
    Wu et al.
    """
    new_traces = np.zeros(traces.shape, dtype=traces.dtype)

    for i in range(len(traces)):
        level = np.random.randint(1, desync_level + 1)
        new_traces[i][:-level] = np.roll(traces[i], -level)[:-level]

    return new_traces


def construct_countermeasure_datasets(dataset_name, desync=50):
    """
    Constructs noisy and desynchronised trace sets for traces with the given
    `dataset_name` and saves the results.
    """
    (x_train, _, _, _, x_atk, _, _, _) = \
        load_data(dataset_name, hw=False, remote=False)

    x_train_noisy = apply_noise(x_train)
    x_atk_noisy = apply_noise(x_atk)
    x_train_desync = apply_desync(x_train, desync_level=desync)
    x_atk_desync = apply_desync(x_atk, desync_level=desync)

    np.save(f"{dataset_name}_train_traces_noisy.npy", x_train_noisy)
    np.save(f"{dataset_name}_atk_traces_noisy.npy", x_atk_noisy)
    np.save(f"{dataset_name}_train_traces_desync{desync}.npy", x_train_desync)
    np.save(f"{dataset_name}_atk_traces_desync{desync}.npy", x_atk_desync)

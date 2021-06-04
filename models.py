import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops.gen_array_ops import split
import numpy as np

from helpers import load_model_weights_from_ga_results, consecutive_int_groups


def build_small_cnn_ascad():
    """
    Constructs and returns the small convolutional NN proposed by Zaid et al.
    to attack the ASCAD data set.
    """
    # Use 4 CONV filters of size 1, SeLU, and 2 FC layers of 10 neurons each
    # The resulting network has 16960 trainable weights
    cnn = keras.Sequential(
        [
            keras.layers.Conv1D(4, 1, activation=tf.nn.selu, padding='same', input_shape=(700,1), kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.BatchNormalization(),
            keras.layers.AveragePooling1D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.Dense(256, activation=tf.nn.softmax)
        ]
    )

    return cnn


def load_small_cnn_ascad(official=False):
    """
    Loads the trained version of the small convolutional NN proposed by Zaid et
    al. directly from an HDF5 file and returns it.

    Arguments:
        official: If True, use the official network taken from the paper's
            corresponding GitHub page.
    """
    path = "./trained_models/efficient_cnn_ascad_model_from_github.h5" \
           if official \
           else "./trained_models/efficient_cnn_ascad_model_17kw.h5"
    return keras.models.load_model(path)


def build_small_cnn_ascad_no_batch_norm(save=False):
    """
    Constructs and returns the small convolutional NN proposed by Zaid et al.
    to attack the ASCAD data set, but leaves out the batch normalisation layer,
    making it less efficient is trained with SGD, but appropriate for a GA.
    """
    # Use 4 CONV filters of size 1, SeLU, and 2 FC layers of 10 neurons each
    # The resulting network has 30944 trainable weights
    cnn = keras.Sequential(
        [
            keras.layers.Conv1D(4, 1, activation=tf.nn.selu, padding='same', input_shape=(700,1), kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.AveragePooling1D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.Dense(256, activation=tf.nn.softmax)
        ]
    )
    
    if save:
        cnn.save("./trained_models/efficient_cnn_ascad_model_17kw_no_bn.h5")

    return cnn


def load_small_cnn_ascad_no_batch_norm():
    """
    Loads an untrained, small convolutional NN with an architecture proposed by
    Zaid et al. directly from an HDF5 file and returns it. This version of the
    model leaves out the batch normalisation layer to make it apt for
    optimisation through a GA.
    """
    path = "./trained_models/efficient_cnn_ascad_model_17kw_no_bn.h5"
    return keras.models.load_model(path, compile=False)


def build_small_mlp_ascad(save=False):
    """
    Constructs and returns the small MLP proposed by Wouters et al. to attack
    the ASCAD data set. This model omits the CONV layer from the model proposed
    by Zaid et al.
    """
    mlp = keras.Sequential(
        [
            keras.layers.AveragePooling1D(pool_size=2, strides=2, input_shape=(700,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu),
            keras.layers.Dense(10, activation=tf.nn.selu),
            keras.layers.Dense(256, activation=tf.nn.softmax)
        ]
    )

    if save:
        mlp.save("./trained_models/efficient_mlp_ascad_model_6kw.h5")

    return mlp


def load_small_mlp_ascad(trained=False):
    """
    Loads an MLP with an architecture proposed by Wouters et al. directly from
    an HDF5 file and returns it.
    """
    path = "./trained_models/efficient_mlp_ascad_model_6kw.h5" if not trained \
        else "./trained_models/official_ascad_mlp_trained.hdf5"
    return keras.models.load_model(path, compile=False)


def small_mlp_cw(build=False, hw=False, n_dense=2):
    """
    Builds or loads and returns an MLP for the ChipWhisperer data set.
    """
    leakage_str = "hw" if hw else "id"
    model_str = f"./trained_models/cw_mlp_untrained_{leakage_str}_{n_dense}.h5"

    if build:
        n_output_classes = 256 if not hw else 9
        mlp = keras.Sequential([
            keras.layers.AveragePooling1D(pool_size=2, strides=2, input_shape=(5000,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(n_dense, activation=tf.nn.selu),
            keras.layers.Dense(n_output_classes, activation=tf.nn.softmax)
        ])
        mlp.save(model_str)
    else:
        return keras.models.load_model(model_str, compile=False)

    return mlp


def small_mlp_cw_func(build=False, hw=False, n_dense=2):
    """
    Builds or loads and returns an MLP for the ChipWhisperer data set.
    """
    leakage_str = "hw" if hw else "id"
    model_str = f"./trained_models/cw_mlp_untrained_{leakage_str}_{n_dense}.h5"
    if build:
        # Act as if we're given a weight vector & indices to mask
        connected_input_idxs = np.concatenate(
            (np.arange(0, 2300), np.arange(2302, 4804), np.arange(4805, 5000))
        )
        input_idx_groups = consecutive_int_groups(connected_input_idxs)

        n_output_classes = 256 if not hw else 9
        inputs = keras.Input(shape=(5000, 1))
        # x = keras.layers.AveragePooling1D(pool_size=2, strides=2)(inputs)
        # x = keras.layers.Flatten()(inputs)  # Output shape (None, 5000)
        # Test splitting functionality
        # split_inputs = [inputs[:, i*2, :] for i in range(2500)]  # TODO: Optimise this part, e.g. with a masking layer instead of splitting

        # Obtain input layer as separate groups of consecutive input values
        input_layers = [
            keras.layers.Flatten()(inputs[:, idxs[0]:(idxs[-1] + 1), :])
            for idxs in input_idx_groups
        ]
        # x = inputs[:, connected_input_idxs, :]  # Desired shape = (None, 1). Strategy to achieve this: filter to (None, n, 1) -> Flatten OR Flatten -> filter from (None, 5000) to (None, n)
        x = keras.layers.Concatenate()(input_layers)
        # x = keras.layers.Lambda(lambda l: l[:, :2500], output_shape=(2500,))(x)
        x = keras.layers.Dense(n_dense, activation=tf.nn.selu, name="dense1")(x)
        x = keras.layers.Dense(n_output_classes, activation=tf.nn.softmax, name="output")(x)
        mlp = keras.Model(inputs, x)
        # mlp.save(model_str)
    else:
        return keras.models.load_model(model_str, compile=False)
    return mlp


def mini_mlp_cw(build=False, hw=True):
    """
    Builds and returns an MLP for the ChipWhisperer data set with fewer than
    2000 weights by stacking 2 pooling layers.
    """
    leakage_str = "hw" if hw else "id"
    model_str = f"./trained_models/mini_cw_mlp_2kw_{leakage_str}.h5"

    if build:
        n_output_classes = 256 if not hw else 9
        mlp = keras.Sequential([
            keras.layers.AveragePooling1D(pool_size=2, strides=2, input_shape=(5000,1)),
            keras.layers.AveragePooling1D(pool_size=2, strides=2, input_shape=(2500,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation=tf.nn.selu),
            keras.layers.Dense(n_output_classes, activation=tf.nn.softmax)
        ])
        mlp.save(model_str)
    else:
        return keras.models.load_model(model_str, compile=False)

    return mlp


def build_small_cnn_ascad_trainable_conv():
    """
    Builds and returns a CNN where only the CONV layer is trainable. Note that
    the batch normalisation layer is "trained" during this process as well.
    """
    cnn = keras.Sequential(
        [
            keras.layers.Conv1D(4, 1, activation=tf.nn.selu, padding='same', input_shape=(700,1), kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.BatchNormalization(),
            keras.layers.AveragePooling1D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform(), trainable=False),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform(), trainable=False),
            keras.layers.Dense(256, activation=tf.nn.softmax, trainable=False)
        ]
    )

    return cnn


def load_small_cnn_ascad_trainable_conv():
    """
    Loads and returns a CNN where only the CONV layer is trainable. Note that
    the model includes a trained batch normalisation layer.
    """
    path = "./trained_models/efficient_cnn_ascad_trained_conv.h5"
    return keras.models.load_model(path, compile=False)


def build_small_mlp_ascad_trainable_first_layer(save=False):
    """
    Builds and returns an MLP where only the first layer is trainable.
    """
    mlp = keras.Sequential(
        [
            keras.layers.AveragePooling1D(pool_size=2, strides=2, input_shape=(700,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu),
            keras.layers.Dense(10, activation=tf.nn.selu, trainable=False),
            keras.layers.Dense(256, activation=tf.nn.softmax, trainable=False)
        ]
    )

    if save:
        mlp.save("./trained_models/efficient_mlp_ascad_model_trainable_first.h5")

    return mlp


def load_small_mlp_ascad_trained_first_layer():
    """
    Loads and returns an MLP where only the first layer was trained with SGD.
    """
    path = "./trained_models/efficient_mlp_ascad_model_trained_first.h5"
    return keras.models.load_model(path, compile=False)


def build_small_cnn_rand_init():
    """
    Constructs and returns the small convolutional NN proposed by Zaid et al.
    to attack the ASCAD data set.
    """
    # Use 4 CONV filters of size 1, SeLU, and 2 FC layers of 10 neurons each
    # The resulting network has 30944 trainable weights
    cnn = keras.Sequential(
        [
            keras.layers.Conv1D(4, 1, activation=tf.nn.selu, padding='same', input_shape=(700,1), kernel_initializer=keras.initializers.random_uniform(-1, 1), name="conv"),
            keras.layers.BatchNormalization(name="batch_norm"),
            keras.layers.AveragePooling1D(pool_size=2, strides=2, name="pool"),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.random_uniform(-1, 1), name="dense0"),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.random_uniform(-1, 1), name="dense1"),
            keras.layers.Dense(256, activation=tf.nn.softmax, name="output")
        ]
    )
    
    return cnn


def load_nn_from_experiment_results(experiment_name, load_model_function):
    """
    Loads an NN with the given model loading function and assigns it the
    weights corresponding to the given GA experiment name.
    """
    weights = load_model_weights_from_ga_results(experiment_name)
    nn = load_model_function()
    nn.set_weights(weights)

    return nn


def set_nn_load_func(nn_str, args=()):
    global NN_LOAD_FUNC
    global NN_LOAD_ARGS

    mapping = {
        "mlp_ascad": load_small_mlp_ascad,
        "cnn_ascad": load_small_cnn_ascad_no_batch_norm,
        "mlp_cw": small_mlp_cw,
        "mini_mlp_cw": mini_mlp_cw
    }

    NN_LOAD_FUNC = mapping[nn_str]
    NN_LOAD_ARGS = args


class MaskedDense(keras.layers.Dense):
    """
    A standard NN layer with the option to mask incoming connections.
    """

    def __init__(self, units, activation, mask, **kwargs):
        """
        Constructs a Dense layer object with `units` neurons and a `mask` array
        where mask values 1 and 0 represent enabled and disabled connections,
        respectively.
        """
        self.mask = mask

        super(MaskedDense, self).__init__(units, activation, **kwargs)

    def call(self, inputs):
        self.kernel = self.kernel * self.mask
        super(MaskedDense, self).call(inputs)

    def weight_count(self):
        """
        Returns the true amount of weights connecting to this layer, computed
        by using the `mask` array provided upon layer construction.
        """
        return np.count_nonzero(self.mask*self.kernel)

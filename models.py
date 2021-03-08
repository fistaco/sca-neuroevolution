import tensorflow as tf
from tensorflow import keras

from helpers import load_model_weights_from_ga_results


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


def load_small_mlp_ascad():
    """
    Loads an untrained MLP with an architecture proposed by Wouters et al.
    directly from an HDF5 file and returns it.
    """
    path = "./trained_models/efficient_mlp_ascad_model_6kw.h5"
    return keras.models.load_model(path, compile=False)


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

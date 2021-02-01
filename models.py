import tensorflow as tf
from tensorflow import keras


def build_small_cnn_ascad():
    """
    Constructs and returns the small convolutional NN proposed by Zaid et al.
    to attack the ASCAD data set.

    Arguments:
        data_input_shape: A tuple assumed to represent the shape of a single
        input sample from the data before it was converted for CNN
        compatibility.
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

    return cnn

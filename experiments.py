import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data_processing import load_ascad_data
from genetic_algorithm import GeneticAlgorithm
from helpers import exec_sca
from models import build_small_cnn
from plotting import plot_gens_vs_fitness


def ga_grid_search():
    pass


def run_ga(max_gens, pop_size, mut_power, mut_rate, crossover_rate,
           mut_power_decay_rate, truncation_proportion, atk_set_size, nn,
           x_validation, y_validation, ptexts_validation, x_test, y_test,
           ptexts_test, true_subkey):
    ga = GeneticAlgorithm(
        max_gens,
        pop_size,
        mut_power,
        mut_rate,
        crossover_rate,
        mut_power_decay_rate,
        truncation_proportion,
        atk_set_size
    )

    # Obtain the best network resulting from the GA
    best_indiv = \
        ga.run(nn, x_validation, y_validation, ptexts_validation, true_subkey)
    best_nn = best_indiv.model

    # Evaluate the best network's performance on the test set
    key_rank = exec_sca(best_nn, x_test, y_test, ptexts_test, true_subkey)
    # TODO: plot generational fitness improvement

    print(f"Key rank on validation set: {best_indiv.fitness}")
    print(f"Key rank on test set: {key_rank}")


def small_cnn_sgd_sca(save=True):
    # Load the ASCAD data set with 700 points per trace
    PATH = "./../ASCAD_data/ASCAD_databases/ASCAD.h5"
    (x, y, x_atk, y_atk, train_meta, atk_meta) = \
        load_ascad_data(PATH, load_metadata=True)
    original_input_shape = (700, 1)
    x_train, y_train = x[:45000], y[:45000]
    x_valid, y_valid = x[:-5000], y[:-5000]

    # Define hyperparameters
    n_epochs = 50
    batch_size = 50
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    # Declare easily accessible variables for relevant metadata
    full_key = atk_meta['key'][0]
    target_subkey = full_key[1]
    atk_ptexts = atk_meta['plaintext']

    # Convert labels to one-hot encoding probabilities
    y_train_converted = keras.utils.to_categorical(y_train, num_classes=256)

    # Reshape the trace input to come in singleton arrays for CNN compatibility
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_atk_reshaped = x_atk.reshape((x_atk.shape[0], x_atk.shape[1], 1))

    # Train CNN
    cnn = build_small_cnn(original_input_shape)
    cnn.compile(optimizer, loss_fn)
    history = cnn.fit(x_train, y_train_converted, batch_size, n_epochs)

    # Save the model if desired
    if save:
        cnn.save('./trained_models/efficient_cnn_ascad_model_31kw.h5')

    # Attack with the trained model
    key_rank = exec_sca(cnn, x_atk_reshaped, y_atk, atk_ptexts, target_subkey, subkey_idx=1)

    print(f"Key rank obtained with efficient CNN on ASCAD: {key_rank}")

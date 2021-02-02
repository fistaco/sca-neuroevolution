import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras

from data_processing import load_ascad_data, train_test_split
from experiments import ga_grid_search, run_ga, small_cnn_sgd_sca
from models import build_small_cnn_ascad
from params import MUTATION_POWER, MUTATION_RATE, CROSSOVER_RATE, \
    MAX_GENERATIONS, POPULATION_SIZE, TRUNCATION_PROPORTION, TOURNAMENT_SIZE, \
    MUTATION_POWER_DECAY, FITNESS_INHERITANCE_DECAY, ATTACK_SET_SIZE


if __name__ == "__main__":
    # small_cnn_sgd_sca(subkey_idx=2)

    (x, y, x_atk, y_atk, train_meta, atk_meta) = \
        load_ascad_data(load_metadata=True)
    original_input_shape = (700, 1)
    x_train, y_train = x[:45000], y[:45000]

    # Declare easily accessible variables for relevant metadata
    subkey_idx = 2
    target_train_subkey = train_meta['key'][0][subkey_idx]
    train_ptexts = train_meta['plaintext']
    target_atk_subkey = atk_meta['key'][0][subkey_idx]
    atk_ptexts = atk_meta['plaintext']

    # Convert labels to one-hot encoding probabilities
    y_train_converted = keras.utils.to_categorical(y_train, num_classes=256)

    # Reshape the trace input to come in singleton arrays for CNN compatibility
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_atk = x_atk.reshape((x_atk.shape[0], x_atk.shape[1], 1))

    # Train the CNN by running it through the GA
    cnn = build_small_cnn_ascad()
    run_ga(5, 2, MUTATION_POWER, MUTATION_RATE,
           CROSSOVER_RATE, MUTATION_POWER_DECAY, TRUNCATION_PROPORTION,
           ATTACK_SET_SIZE, cnn, x_train, y_train, train_ptexts, x_atk, y_atk,
           atk_ptexts, target_atk_subkey, parallelise=True)

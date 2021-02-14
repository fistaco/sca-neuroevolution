import matplotlib.pyplot as plt
import numpy as np


def plot_gens_vs_fitness(fitnesses_per_gen, experiment_name):
    """
    Constructs and saves a plot with the generations on the x-axis and fitness
    values on the y-axis.

    Arguments:
        fitnesses_per_gen: A 1-dimensional array of variable length
        containing the fitness values.

        experiment_name: The name of the results' corresponding experiment.
    """
    plt.title(f"Generations ~ fitness ({experiment_name})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness (key rank)")
    plt.plot(np.arange(len(fitnesses_per_gen)), fitnesses_per_gen)
    plt.grid(True)
    plt.savefig(f"./fig/{experiment_name}_gens_vs_fitness.png")
    plt.clf()


def plot_n_traces_vs_key_rank(key_ranks, experiment_name):
    """
    Constructs and saves a plot with the amount of traces on the x-axis and the
    (mean) key rank on the y-axis.

    Arguments:
        rank_per_trace_num: A dictionary with the trace amounts as keys and the
        corresponding (mean) key ranks as values.

        experiment_name: The name of the results' corresponding experiment.
    """
    plt.title(f"Amount of traces ~ key rank ({experiment_name})")
    plt.xlabel("Traces")
    plt.ylabel("Key rank")
    plt.plot(np.arange(len(key_ranks)), key_ranks)
    plt.grid(True)
    plt.savefig(f"./fig/{experiment_name}_traces_vs_keyrank.png")
    plt.clf()

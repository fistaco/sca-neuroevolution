import matplotlib.pyplot as plt
import numpy as np


def plot_gens_vs_fitness(experiment_name, fitnesses_per_gen):
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
    plt.ylabel("Fitness")
    plt.plot(np.arange(len(fitnesses_per_gen)), fitnesses_per_gen)
    plt.ylim(-1, 180)
    plt.grid(True)
    plt.savefig(f"./fig/{experiment_name}_gens_vs_fitness.png")
    plt.clf()


def plot_n_traces_vs_key_rank(experiment_name, *key_rankss, labels=None):
    """
    Constructs and saves a plot with the amount of traces on the x-axis and the
    (mean) key rank on the y-axis.

    Arguments:
        experiment_name: The name of the results' corresponding experiment.

        key_rankss: A list in which each item is a numpy array containing the
        (mean) key rank for each trace amount.

        labels: A list of labels corresponding to the lists of key ranks.
    """
    plt.title(f"Amount of traces ~ key rank ({experiment_name})")
    plt.xlabel("Traces")
    plt.ylabel("Key rank")

    for key_ranks in key_rankss:
        plt.plot(np.arange(len(key_ranks)), key_ranks)
    plt.ylim(0, 180)
    plt.grid(True)
    if labels:
        plt.legend(labels)

    plt.savefig(f"./fig/{experiment_name}_traces_vs_keyrank.png")
    plt.clf()

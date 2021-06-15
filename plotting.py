import matplotlib.pyplot as plt
from matplotlib import cm
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
    # plt.ylim(-1, 180)
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


def plot_var_vs_key_rank(var_values, key_ranks, result_category, box=False,
                         eval_fitness=False):
    """
    Plots a given list of variable values against a given list of key ranks,
    while labelling the variable according to a given result category.
    """
    var_name = result_category.name.replace("_", " ").capitalize()
    eval_metric_name = "fitness" if eval_fitness else "incremental key rank"

    uniq_vals = np.unique(var_values)
    n = len(uniq_vals)
    inc_kr_groups = [key_ranks[i:i+5] for i in range(0, len(key_ranks), 5)]
    mean_key_ranks = [np.mean(group) for group in inc_kr_groups]
    yerr = [np.std(group) for group in inc_kr_groups]

    plt.title(f"{var_name} ~ final {eval_metric_name}")
    plt.xlabel(var_name)
    plt.ylabel(eval_metric_name.capitalize())
    plt.grid(True)

    if box:
        plt.boxplot(inc_kr_groups, labels=uniq_vals)
    else:
        plt.errorbar(uniq_vals, mean_key_ranks, yerr)

    suffix = "fit" if eval_fitness else "inckr"
    plt.savefig(f"./fig/{result_category.name.lower()}-vs-{suffix}.png")
    plt.clf()


def plot_2d(xs, ys, x_label, y_label, title):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(xs, ys)
    plt.grid(True)
    plt.savefig(f"./fig/{title}.png")
    plt.clf()


def plot_3d(xs, ys, zs, x_label, y_label, z_label, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(title)
    x_mg, y_mg = np.meshgrid(xs, ys)
    # ax.plot_wireframe(x_mg, y_mg, zs)
    surf = ax.plot_surface(x_mg, y_mg, zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    plt.savefig(f"./fig/{title}.png")
    plt.close()

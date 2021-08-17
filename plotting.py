import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

from helpers import first_zero_value_idx


def plot_gens_vs_fitness(experiment_name, *fit_progress_lists, labels=None):
    """
    Constructs and saves a plot with the generations on the x-axis and fitness
    values on the y-axis.

    Arguments:
        fitnesses_per_gen: A 1-dimensional array of variable length
        containing the fitness values.

        experiment_name: The name of the results' corresponding experiment.
    """
    plt.title(f"Generations ~ fitness ({experiment_name.replace('_', ' ')})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    # max_gens = max([len(a) for a in fit_progress_lists])
    for fitnesses_per_gen in fit_progress_lists:
        fits = np.array(fitnesses_per_gen)
        close_to_zero_idxs = np.isclose(fits, np.zeros(len(fits)), atol=1e-4)
        fits[close_to_zero_idxs] = 0        

        nonzero_fits = fits[fits != 0]
        fits[len(nonzero_fits):] = nonzero_fits[-1]

        # fit_progress = np.zeros(max_gens)
        # fit_progress[:len_nz] = fitnesses_per_gen[:len_nz]
        # fit_progress[len_nz:] = nonzero_fits[-1]

        plt.plot(np.arange(len(fits)), fits)

    if labels:
        plt.legend(labels)

    plt.grid(True)
    plt.savefig(f"./fig/{experiment_name}_gens_vs_fitness.png")
    plt.clf()


def plot_n_traces_vs_key_rank(exp_name, *key_rankss, labels=None):
    """
    Constructs and saves a plot with the amount of traces on the x-axis and the
    (mean) key rank on the y-axis.

    Arguments:
        experiment_name: The name of the results' corresponding experiment.

        key_rankss: A list in which each item is a numpy array containing the
        (mean) key rank for each trace amount.

        labels: A list of labels corresponding to the lists of key ranks.
    """
    plt.title(f"Amount of traces ~ key rank ({exp_name})")
    plt.xlabel("Traces")
    plt.ylabel("Key rank")
    
    xlim_max = -1

    for key_ranks in key_rankss:
        # xlim_max should be the approx. the last idx where kr 0 is achieved
        cutoff_idx = first_zero_value_idx(key_ranks) + 50
        xlim_max = max(xlim_max, cutoff_idx)

        plt.plot(np.arange(len(key_ranks)), key_ranks)

    plt.xlim(0, min(xlim_max, len(key_rankss[0])))
    plt.ylim(0, 180)
    plt.grid(True)
    if labels:
        plt.legend(labels)

    formatted_exp_name = exp_name.replace(" ", "_")
    plt.savefig(f"./fig/{formatted_exp_name}_traces_vs_keyrank.png")
    plt.clf()


def plot_var_vs_key_rank(var_values, key_ranks, result_category=None,
                         box=False, eval_fitness=False, var_name=None):
    """
    Plots a given list of variable values against a given list of key ranks,
    while labelling the variable according to a given result category.
    """
    var_name = result_category.name.replace("_", " ").capitalize() \
        if result_category is not None else var_name
    eval_metric_name = "fitness" if eval_fitness else "incremental key rank"

    _, uniq_idxs = np.unique(var_values, return_index=True)
    uniq_vals = var_values[np.sort(uniq_idxs)]
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
    plt.savefig(f"./fig/{var_name.replace(' ', '_').lower()}-vs-{suffix}.png")
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
    surf = ax.plot_surface(
        x_mg, y_mg, zs, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    plt.savefig(f"./fig/{title}.png")
    plt.close()


def nn_weights_heatmaps(weightss, exp_label):
    """
    Plots heatmaps for each layer of the given `weights` array, which should be
    formatted according to Keras NN weight formatting.
    """
    vmin = np.min([np.min(ws) for ws in weightss])
    vmax = np.max([np.max(ws) for ws in weightss])

    for i in range(0, len(weightss), 2):
        weights = reshape_to_2d_singleton_array(weightss[i])
        biases = reshape_to_2d_singleton_array(weightss[i + 1])   

        nn_layer_heatmap(
            weights, f"{exp_label}_heatmap_weights_{i//2}", vmin, vmax
        )
        nn_layer_heatmap(
            biases, f"{exp_label}_heatmap_biases_{i//2}", vmin, vmax
        )


def nn_layer_heatmap(layer, filename, vmin, vmax):
    """
    Plots the heatmap for the given array of weights or biases under the given
    filename.
    """
    sns.heatmap(layer, vmin=vmin, vmax=vmax, xticklabels=False)
    plt.savefig(f"{filename}.png")
    # plt.savefig(f"fig/{filename}.png")
    plt.clf()

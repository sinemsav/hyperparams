from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import baseline_results, mode, compute_accuracy, mae, std
from read_data import get_local_res, get_federated_res
from heuristic_funcs import aggregate_results
from constants import HEUR_DICT, HP_GRID


def get_heuristic_errors(dataset_name, skews, nrs_parties, type_of_skew, hp_name, versions=(1, 2, 3), plot=True):
    """ Function returning the difference between hyperparameters from federated grid search and the heuristic functions
    and baseline functions.

    Args:
        dataset_name (str): Name of the dataset.
        skews (list): List of distribution skew values.
        nrs_parties (list): List of number of clients.
        type_of_skew (str): Type of distribution skew.
        hp_name (str): Name of hyperparameter.
        versions (tuple, optional): Versions of heuristic functions. Defaults to (1, 2, 3).
        plot (bool, optional): If plot True, display boxplots of the heuristic
        function outputs. Defaults to True.

    Returns:
        agg_errors (dict): Dictionary with version of heuristic function as key and error as value.
        X (list): List of list containing local client hyperparameters.
        y (list): Optimal hyperparameters from federated experiment results.
        y_hat (list): Returned hyperparameters from heuristic function.
    """
    agg_errors = defaultdict(list)
    X = []
    y = []
    y_hat = []

    hp_in = HEUR_DICT[hp_name]["in"]
    hp_out = HEUR_DICT[hp_name]["out"]

    if plot:
        _, ax = plt.subplots(nrows=len(skews), ncols=len(
            nrs_parties), figsize=(10, 20), sharey=True)

    for i, s in enumerate(skews):
        for j, p in enumerate(nrs_parties):
            # Get results of individual clients
            hps, accs, _, ratios = get_local_res(
                dataset_name, s, p, type_of_skew, hp_name=hp_in
            )

            # Calculate global hyperparameter using heuristics
            heuristic_res = [
                aggregate_results(
                    hps, accs, ratios, type_of_skew, hp_name, v=v
                )[hp_out]
                for v in versions
            ]

            # Baselines
            mean_params = baseline_results(hps, np.mean, hp_name)
            median_params = baseline_results(hps, np.median, hp_name)
            mode_params = baseline_results(hps, mode, hp_name)

            mean_res = mean_params[hp_out]
            median_res = median_params[hp_out]
            mode_res = mode_params[hp_out]

            # Get FEDAVG results (ground truth)
            fedavg_params = get_federated_res(dataset_name, s, p, type_of_skew)
            fedavg_res = fedavg_params[hp_out]

            y_hat.append(heuristic_res)
            y.append(fedavg_res)
            X.append(hps[hp_in])

            for v in versions:
                agg_errors[v].append(heuristic_res[v] - fedavg_res)

            # Calculate errors
            agg_errors["mean"].append(mean_res - fedavg_res)
            agg_errors["median"].append(median_res - fedavg_res)
            agg_errors["mode"].append(mode_res - fedavg_res)

            # Boxplots
            if plot:
                sns.boxplot(y=hps[hp_in], showmeans=False, ax=ax[i, j])
                sns.swarmplot(y=hps[hp_in], color=".25", ax=ax[i, j])

                for v in versions:
                    ax[i, j].scatter(x=0, y=heuristic_res[v], s=100,
                                     label=f"heuristic v{v}")
                ax[i, j].scatter(x=0, y=fedavg_res, s=100, marker="x",
                                 color="yellow", label="fedavg")

                if hp_name == "bs":
                    ax[i, j].set_yscale("log", base=2)

                ax[i, j].set_title(
                    f"{len(ratios)} clients, {type_of_skew} skew {s}, {dataset_name}")
                ax[i, j].set_xlabel("Number of clients")
                ax[i, j].set_ylabel(hp_name)
                ax[i, j].legend()

    return agg_errors, (X, y, y_hat)


def print_heur_results(agg_es, X, y, y_hat, hp_name, detailed=False):
    """ Display output of baselines and heuristic functions compared to the
    optimal grid search hyperparameters.

    Args:
        agg_errors (dict): Dictionary with version of heuristic function as key and error as value.
        X (list): List of list containing local client hyperparameters.
        y (list): Optimal hyperparameters from federated experiment results.
        y_hat (list): Returned hyperparameters from heuristic function.
        hp_name (str): Name of hyperparameter.
        detailed (bool, optional): If True, print inputs (X) and outputs (y, y_hat). Defaults to False.
    """

    print("HEURISTIC RESULTS")

    accs = compute_accuracy(y, y_hat, HP_GRID[hp_name])

    for v in range(len(y_hat[0])):
        print(
            f"Heuristic {v} MAE: {mae(agg_es[v]):.3f}"
            f" ± {std(agg_es[v]):.3f}"
            f" (Bias: {np.mean(agg_es[v]):+.3f})")
        print(f"Heuristic {v} accuracy: {accs[v]}")

    print("BASELINE RESULTS")

    print(
        f"Mean MAE: {mae(agg_es['mean']):.3f}"
        f" ± {std(agg_es['mean']):.3f}"
        f" (Bias: {np.mean(agg_es['mean']):+.3f})")
    print(
        f"Median MAE: {mae(agg_es['median']):.3f}"
        f" ± {std(agg_es['median']):.3f}"
        f" (Bias: {np.mean(agg_es['median']):+.3f})")
    print(
        f"Mode MAE: {mae(agg_es['mode']):.3f}"
        f" ± {std(agg_es['mode']):.3f}"
        f" (Bias: {np.mean(agg_es['mode']):+.3f})")

    if detailed:
        print()
        for i in range(len(X)):
            print("X", X[i])
            print("y", y[i])
            print("y_hat", y_hat[i])

    print(
        f"(mean: {np.mean(y):.3f}, variance: {np.var(y):.3f})")

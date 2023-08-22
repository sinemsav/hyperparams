import numpy as np

from utils import closest_power, filter_outliers
from constants import HEUR_DICT


def lr_heuristic(ratios, nr_parties, hp_values, val_accs, type_of_skew, v=0):
    """ Learning rate heuristic function.

    Args:
        ratios (list): List of ratio of data samples of every client.
        nr_parties (int): Number of clients.
        hp_values (list): List of optimal local learning rates of every client.
        val_accs (list): List of local validation accuracy of every client
        type_of_skew (str): Type of distribution skew of data.
        v (int, optional): Version of heuristic function. Defaults to 0.

    Returns:
        float: Global learning rate returned by heuristic function.
    """
    agg_lr = 0
    lrs = hp_values
    if v == 0:
        # constant
        agg_lr = 0.5
    elif v == 1:
        for i in range(nr_parties):
            agg_lr += ratios[i] / np.sum(ratios) * lrs[i] * (1 - lrs[i])

        # add bias
        agg_lr += .3

    elif v == 2:
        filtered_lrs = filter_outliers(lrs, scale=1.5)
        if type_of_skew == "qty":
            for i in range(nr_parties):
                agg_lr = 0.8 * np.max(filtered_lrs)
        else:
            agg_lr = np.max(filtered_lrs)

    elif v == 3:
        if type_of_skew == "qty":
            agg_lr = np.mean(lrs)
        else:
            for i in range(nr_parties):
                agg_lr += 1.5 * val_accs[i] / np.sum(val_accs) * lrs[i]

    return agg_lr


def momentum_heuristic(ratios, nr_parties, hp_values, val_accs, type_of_skew, v=0):
    """ Momentum heuristic function.

    Args:
        ratios (list): List of ratio of data samples of every client.
        nr_parties (int): Number of clients.
        hp_values (list): List of optimal local momentums of every client.
        val_accs (list): List of local validation accuracy of every client
        type_of_skew (str): Type of distribution skew of data.
        v (int, optional): Version of heuristic function. Defaults to 0.

    Returns:
        float: Global momentum returned by heuristic function.
    """
    agg_mom = 0
    moms = hp_values
    if type_of_skew == "feature":
        if v == 0:
            # constant (mean server hp)
            agg_mom = 0.712
        if v == 1:
            for i in range(nr_parties):
                agg_mom += moms[i] * (1 - moms[i]) / nr_parties

            # add bias
            agg_mom += 0.627

        elif v == 2:
            agg_mom = np.percentile(moms, q=20)

        elif v == 3:
            for i in range(nr_parties):
                agg_mom += 0.8 * val_accs[i] / np.sum(val_accs) * moms[i]

    if type_of_skew == "label":
        if v == 0:
            # constant (mean server hp)
            agg_mom = .763
        if v == 1:
            for i in range(nr_parties):
                agg_mom += moms[i] * (1 - moms[i]) / nr_parties

            # add bias
            agg_mom += .676

        elif v == 2:
            agg_mom = np.percentile(moms, q=40)

        elif v == 3:
            for i in range(nr_parties):
                agg_mom += val_accs[i] / np.sum(val_accs) * moms[i]

    if type_of_skew == "qty":
        if v == 0:
            # constant (mean server hp)
            agg_mom = .412
        if v == 1:
            for i in range(nr_parties):
                agg_mom += ratios[i] / np.sum(ratios) * moms[i]

            agg_mom -= .37

        elif v == 2:
            for i in range(nr_parties):
                agg_mom = nr_parties / (20 * (np.median(moms) + 1))

        elif v == 3:
            for i in range(nr_parties):
                agg_mom += 0.5 * val_accs[i] / np.sum(val_accs) * moms[i]

    return agg_mom


def batch_size_heuristic(ratios, nr_parties, hp_values, val_accs, type_of_skew, v=0):
    """ Batch size heuristic function.

    Args:
        ratios (list): List of ratio of data samples of every client.
        nr_parties (int): Number of clients.
        hp_values (list): List of optimal local batch sizes of every client.
        val_accs (list): List of local validation accuracy of every client
        type_of_skew (str): Type of distribution skew of data.
        v (int, optional): Version of heuristic function. Defaults to 0.

    Returns:
        float: Batch size returned by heuristic function.
    """
    agg_bs = 0
    batch_sizes = hp_values
    if type_of_skew == "feature":
        if v == 0:
            # constant (median fedavg batch size)
            agg_bs = 8
        if v == 1:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] * val_accs[i] / (2 * np.sum(val_accs))

        elif v in [2, 3]:
            fil_bs = filter_outliers(batch_sizes)
            agg_bs = np.min(fil_bs) / 2

    elif type_of_skew == "label":
        if v == 0:
            # constant (median fedavg batch size)
            agg_bs = 8

        if v == 1:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] * val_accs[i] / (2 * np.sum(val_accs))

        if v in [2, 3]:
            fil_bs = filter_outliers(batch_sizes)
            agg_bs = np.min(fil_bs) / 2

    elif type_of_skew == "qty":
        if v == 0:
            # constant (median fedavg batch size)
            agg_bs = 8

        if v == 1:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] * val_accs[i] / (2 * np.sum(val_accs))

        if v in [2, 3]:
            fil_bs = filter_outliers(batch_sizes)
            agg_bs = np.min(fil_bs) / 2

    return closest_power(agg_bs)


def aggregate_results(hps, accs, ratios, type_of_skew, hp_name=None, v=0):
    """ Function performing heuristic functions for a single or all
    hyperparameters.

    Args:
        hps (list): List of optimal local hyperparameters for each client.
        accs (list): List of local validation accuracy for each client.
        ratios (list): List of data sample ratios for each client.
        type_of_skew (str): Type of distribution skew of data.
        hp_name (str, optional): Name of hyperparameter. Defaults to None.
        v (int, optional): Version of heuristic function. Defaults to 0.

    Returns:
        dict: Dictionary with hyperparameter name as key and hyperparameter value returned by heuristic functions
        as value.
    """
    nr_parties = len(ratios)

    heuristic_fns = dict(
        zip(["lr", "mom", "bs"], [lr_heuristic, momentum_heuristic, batch_size_heuristic]))

    agg_params = dict()
    # Only perform heuristic function for a single hyperparameter
    if hp_name:
        hp_in = HEUR_DICT[hp_name]["in"]
        hp_out = HEUR_DICT[hp_name]["out"]
        agg_params[hp_out] = \
            heuristic_fns[hp_name](ratios=ratios,
                                   nr_parties=nr_parties,
                                   hp_values=hps[hp_in],
                                   val_accs=accs,
                                   type_of_skew=type_of_skew,
                                   v=v)
    # Perform heuristic function for all analyzed hyperparameters
    else:
        for hp, heur_fn in heuristic_fns.items():
            hp_in = HEUR_DICT[hp]["in"]
            hp_out = HEUR_DICT[hp]["out"]
            agg_params[HEUR_DICT[hp]["out"]] = \
                heur_fn(ratios=ratios, nr_parties=nr_parties, hp_values=hps[hp_in],
                        val_accs=accs, type_of_skew=type_of_skew, v=v)

    return agg_params

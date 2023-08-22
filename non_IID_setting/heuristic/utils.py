from math import log, ceil, floor
from collections import Counter

import numpy as np

from constants import HEUR_DICT, INPUT_HEUR, OUTPUT_HEUR


def sum_dicts(d1, d2):
    """ Compute sum of lists of two dictionaries containing lists.

    Args:
        d1 (dict): dictionary containining lists
        d2 (dict): dictionary containing lists

    Returns:
        dict: dictionary containing the sum of lists per key.
    """
    return {k: d1.get(k, []) + d2.get(k, []) for k in set(d1) | set(d2)}


def mode(lst):
    """ Return most common element (mode) of a list.

    Args:
        lst (list): List of values of type t.

    Returns:
        t: Value appearing most often in list.
    """
    c = Counter(lst)
    return c.most_common(1)[0][0]


def closest_power(x, pow=2):
    """ Project float x to closest power of pow.

    Args:
        x (float): Float x.
        pow (int, optional): Integer greater or equal than 2. Defaults to 2.

    Returns:
        int: Power of pow leading to the closest distance to x.
    """
    possible_results = floor(log(x, pow)), ceil(log(x, pow))
    return pow ** min(possible_results, key=lambda z: abs(x - pow ** z))


def closest(x, lst):
    """ Return closest element to x from list lst.

    Args:
        x (number): Numerical value.
        lst (list): List containing numerical values.

    Returns:
        number: Closest element to x from list.
    """
    closest_idx = np.abs(x - np.asarray(lst)).argmin()
    return lst[closest_idx]


def baseline_results(hps, fn, hp_name=None):
    """ Compute baseline function results.

    Args:
        hps (dict): Dictionary with hyperparameter names as key and
        list of local client hyperparameters as value.
        fn (function): Baseline function.
        hp_name (str, optional): Name of hyperparameter. Defaults to None.

    Returns:
        dict: Dictionary containing outputs of baseline function.
    """
    baseline_params = dict()
    if hp_name:
        hp_in = HEUR_DICT[hp_name]["in"]
        hp_out = HEUR_DICT[hp_name]["out"]
        baseline_params[hp_out] = fn(hps[hp_in])
    else:
        for hp_in, hp_out in zip(INPUT_HEUR, OUTPUT_HEUR):
            baseline_params[hp_out] = fn(hps[hp_in])
    return baseline_params


def filter_outliers(lst, scale=1.5):
    """ Filter outliers of list according to Interquartile Range (IQR) method.

    Args:
        lst (list): List containing numerical values
        scale (float, optional): Factor by which a value can be outsied of iqr
                                 interval. Defaults to 1.5.

    Returns:
        list: List of filtered values (without outliers)
    """
    q1 = np.percentile(lst, 25)
    q3 = np.percentile(lst, 75)
    iqr = q3 - q1

    min_lim = q1 - scale * iqr
    max_lim = q3 + scale * iqr

    return [x for x in lst if min_lim <= x <= max_lim]


def compute_accuracy(y, y_hat, grid):
    """ Compute accuracy of heuristic function. All predicted values (y_hat) are
    first projected to the closest value from grid, and then compared to ground
    truth y.

    Args:
        y (list): Ground truth.
        y_hat (list): Returned value.
        grid (list): Experiment hyperparameter grid.

    Returns:
        np.array: Accuracy values for every heuristic function.
    """
    N = len(y_hat)
    y_hat_proj = np.zeros_like(y_hat)

    for i in range(len(y_hat)):
        for j in range(len(y_hat[0])):
            # Project to closest value from grid
            y_hat_proj[i, j] = closest(y_hat[i][j], grid)

    # Calcuate accuracy
    accs = np.sum((y_hat_proj.T - y) == 0, axis=1) / N

    return accs


def mae(es):
    """ Return mean absolute error.

    Args:
        es (list): List of errors (difference between ground truth and returned value)

    Returns:
        float: Mean absolute error value.
    """
    return np.mean(np.abs(es))


def std(es):
    """ Return standard deviation of absolute errors.

    Args:
        es (list): List of errors (difference between ground truth and returned value)

    Returns:
        float: Standard deviation of absolute errors.
    """
    return np.std(np.abs(es))

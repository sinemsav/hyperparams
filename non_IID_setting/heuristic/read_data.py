from collections import defaultdict
import json
import ast

import numpy as np
import pandas as pd

from constants import INPUT_HEUR, DATASETS

RESULTS_PATH = "../results_aggregation"


def get_local_res(dataset_name, skew, nr_parties, type_of_skew, hp_name=None):
    """ Read optimal client hyperparameters from local experiments.

    Args:
        dataset_name (str): Name of dataset.
        skew (float): Value of distribution skew.
        nr_parties (int): Number of clients.
        type_of_skew (str): Type of distribution skew.
        hp_name (str, optional): Name of hyperparameter. Defaults to None.

    Returns:
        best_hp (list): List of optimal local client hyperparameter values.
        accuracies (list): List of optimal local client validation accuracies.
        best_acc (float): Maximal validation accuracy among clients.
        arr_ratios (list): List of local data sample ratios.
    """
    experiment_directory = (f"{RESULTS_PATH}/non-IID_res/{type_of_skew}_skew/"
                            f"{dataset_name.upper()}_non-IID_{type_of_skew}_skew/"
                            f"{nr_parties}_parties")

    if type_of_skew == "qty":
        distributions = []
        with open(
                f"{experiment_directory}/individual/{skew}/{dataset_name}_qty_skew_"
                f"{skew}_{nr_parties}clients_distribution.txt", "r") as reader:
            for i in range(nr_parties):
                distributions.append(reader.readline())
        ratios = []
    else:
        distributions = None
        ratios = []

    accuracies = []

    best_hp = defaultdict(list)

    # Get individual data
    for i in range(nr_parties):
        file_path = (f"{experiment_directory}/individual/{skew}/{skew}_"
                     f"{type_of_skew}_{nr_parties}clts_clt{i}.txt")
        try:
            client_result = pd.read_csv(file_path)
            if type_of_skew == "qty":
                ratio = float(distributions[i].split(
                    ',')[2].replace(' percentage :', ''))
                ratios.append(ratio)
            else:
                ratios.append(1)
            if hp_name:
                best_hp[hp_name].append(client_result.head(
                    1).get([hp_name]).values[0][0])
            else:
                for hp in INPUT_HEUR:
                    best_hp[hp].append(
                        np.array(client_result.head(1).get([hp]).values[0][0]))

            accuracies.append(client_result.head(
                1).get(['val_accuracy']).values[0][0])

        except FileNotFoundError as e:
            print(
                f"File for client {i} in {dataset_name} ({skew}, {nr_parties}, "
                f"{type_of_skew}) does not exist.")
            # raise e

    best_acc = np.max(accuracies)

    if type_of_skew != "qty":
        arr_ratios = np.array(ratios) / len(ratios)
    else:
        arr_ratios = np.array(ratios)

    return best_hp, accuracies, best_acc, list(arr_ratios)


def get_federated_res(dataset_name, skew, nr_parties, type_of_skew):
    """ Read optimal hyperparameter configuration of federated grid search
    on particular distribution skew setting.

    Args:
        dataset_name (str): Name of dataset
        skew (float): Distribution skew parameter value.
        nr_parties (int):  Number of clients.
        type_of_skew (str): Type of distribution skew.

    Returns:
        dict: Dictionary containing optimal hyperparameter configuration.
    """
    experiment_directory = f"{RESULTS_PATH}/fed_grid_search_results"

    # Get FEDAVG data
    with open(f"{experiment_directory}/{dataset_name.lower()}_{type_of_skew}_skew_"
              f"{skew}_{nr_parties}clients.txt", "r") as reader:
        line = reader.readline()

        while not line.startswith("\'client_lr\':"):
            line = line[1:]
        fedavg_data = ast.literal_eval("{" + line[:-2])

    return fedavg_data


def get_federated_val_acc(dataset_name, skew, nr_parties, type_of_skew):
    """ Read best validation accuracy of federated grid search experiment for specific setting.

    Args:
        dataset_name (str): Name of dataset
        skew (float): Distribution skew parameter value
        nr_parties (int): Number of clients
        type_of_skew (str): Type of distribution skew.

    Returns:
        float: Validation accuracy of optimal federated grid search global hyperparameter configuration.
    """
    experiment_directory = f"{RESULTS_PATH}/fed_grid_search_results"

    # Get federated grid search data
    with open(f"{experiment_directory}/{dataset_name.lower()}_{type_of_skew}_skew_"
              f"{skew}_{nr_parties}clients.txt", "r") as reader:
        line = reader.readline()
        # Read only first line, since results sorted in decreasing order
        fedavg_data = ast.literal_eval(line)

    val_acc = fedavg_data[0]
    return val_acc


def load_heur_results(v, melt=True):
    """ Read heuristic function evaluation results into dataframe

    Args:
        v (int): Version of heuristic function.
        melt (bool, optional): If True, unpivot DataFrame from wide to long format. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing heuristic function evaluation results.
    """
    heur_results = {}

    for dataset in DATASETS:
        with open(f'heur_results/heur_{dataset}.json') as f:
            heur_results[dataset] = json.load(f)

    df = pd.concat(objs=(
        pd.DataFrame.from_records(heur_results[d][f"{v}"])
        for d in DATASETS
    ),
        ignore_index=True
    )

    df.rename(columns={"fedavg_acc": "grid_search_acc"}, inplace=True)

    if melt:
        df = pd.melt(df, id_vars=set(df.columns).difference(
            ["grid_search_acc", "heur_acc"]),
            value_name="acc", var_name="res_type")

    return df

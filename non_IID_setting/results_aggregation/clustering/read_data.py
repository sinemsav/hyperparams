"""
code copied from heuristics folder for easier referencing
"""
import ast


RESULTS_PATH = "../"


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
    with open(
        f"{experiment_directory}/{dataset_name.lower()}_{type_of_skew}_skew_"
        f"{skew}_{nr_parties}clients.txt",
        "r",
    ) as reader:
        line = reader.readline()

        while not line.startswith("'client_lr':"):
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
    with open(
        f"{experiment_directory}/{dataset_name.lower()}_{type_of_skew}_skew_"
        f"{skew}_{nr_parties}clients.txt",
        "r",
    ) as reader:
        line = reader.readline()
        # Read only first line, since results sorted in decreasing order
        fedavg_data = ast.literal_eval(line)

    val_acc = fedavg_data[0]
    return val_acc

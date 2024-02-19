"""
This python script is used to extract results (validation accuracies for derived parameters) based on .csv files.


adjustable parameters:

EXPORT_PATH - path/folder where to save extracted results
CSV_FOLDER - folder with the csv files to be used
RETENTION_PERCENTILE - top % of (HP, acc) pairs to be kept by each client individually
MINIMAL_LOCAL_ACCURACY - removing all (HP, acc) pairs that do not reach MINIMAL_LOCAL_ACCURACY
SCORING_FUNCTION - scoring function to be used when ranking clusters
SCORIRNG_TOLERANCE - tolerance to be used when choosing optimal cluster


Script is run for all datasets, skews, nr_clients combinations.
"""
import sys
sys.path.append("../../")

from clustering import clustering, closest_gt_to_point, get_derived_params
from scoring_funcs import scoring_function
from read_data import get_federated_res, get_federated_val_acc
from constants import DATASETS, SKEWS, NR_PARTIES

import json
import os

CSV_FOLDER = "experiments/extended_data"             # folder contains all .csv files for different datasets/skews
EXPORT_PATH = "results/extended_data"                # where to save the results
if not os.path.exists(f"{EXPORT_PATH}/"):
    os.mkdir(f"{EXPORT_PATH}/")

RETENTION_PERCENTILE = 0.15
MINIMAL_LOCAL_ACCURACY = 0.4
SCORING_FUNCTION = 'mean'
SCORING_TOLERANCE = 0.05

for dataset_name in DATASETS:
    for skew_type in SKEWS.keys():
        results = []
        print(f"DATASET: {dataset_name}, SKEW_TYPE: {skew_type}")
        for nr_clients in NR_PARTIES:
            for skew in SKEWS[skew_type]:
                print(f"{nr_clients} clients, skew={skew}")

                fedavg_acc = get_federated_val_acc(
                    dataset_name, skew, nr_clients, skew_type
                )
                res = get_federated_res(dataset_name, skew, nr_clients, skew_type)

                dataset_file_name = f"{CSV_FOLDER}/{dataset_name}_{skew_type}_skew_{skew}_{nr_clients}clients"

                serverData, clientData, clusterFn, clusterFn3d = clustering(
                    dataset_file_name,
                    useDbscan=True,
                    shouldPrint=False,
                    drawFig=False,
                    saveFig=False,
                    top_client_acc_percentile=RETENTION_PERCENTILE,
                    further_restrict=True,
                    min_local_accuracy=MINIMAL_LOCAL_ACCURACY,
                )

                ind_list = scoring_function(
                    clientData,
                    clusterFn,
                    func_name=SCORING_FUNCTION,
                    shouldPrint=False,
                    tolerance=SCORING_TOLERANCE,
                )
                ind_list3d = scoring_function(
                    clientData,
                    clusterFn3d,
                    func_name=SCORING_FUNCTION,
                    shouldPrint=False,
                    tolerance=SCORING_TOLERANCE,
                )

                derived_params = get_derived_params(
                    ind_list, clientData, dataset_file_name, fedavg_acc
                )
                derived_params3d = get_derived_params(
                    ind_list3d, clientData, dataset_file_name, fedavg_acc
                )

                results.append(
                    {
                        **{
                            "dataset": dataset_name,
                            "nr_clients": nr_clients,
                            "skew_type": skew_type,
                            "skew": skew,
                            "global_acc": fedavg_acc,
                            "global_lr_mom_bs_acc": (
                                res["server_lr"],
                                res["server_momentum"],
                                res["batch_size"],
                                fedavg_acc,
                            ),
                            "derived_params": derived_params,
                            "derived_params3d": derived_params3d,
                        }
                    }
                )

        file_name = f"{EXPORT_PATH}/{dataset_name}_{skew_type}.json"

        with open(file_name, "w") as outfile:
            json.dump(results, outfile)

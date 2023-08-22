"""
Script used for clustering client HPs in 2d and 3d.
Preprocessing is applied.

DATA_FOLDER - folder with the csv files to be analyzed

To run the script uncomment last two lines.
"""
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, OPTICS

from agg_funcs import average_params, average_params_weighted_acc


def get_gt_data(dataset_name):
    """Get ground truth data based on dataset_name parameter.
    Dataset name is of form: "{folder}/{dataset}_{skew_type}_skew_{skew}_{nr_clients}clients"

    Args:
        dataset_name (str): path to the dataset/experiment

    Returns:
        pandas.DataFrame: dataframe containing the ground truth experiment data
    """
    if ".csv" not in dataset_name:
        dataset_name = dataset_name + ".csv"

    data = pd.read_csv(dataset_name, sep=",", header="infer")

    # ground truth data discovered by federated grid search
    gt = data[(data.clientID == -1)]
    gt = gt.sort_values(by="accuracy", ascending=False)

    return gt


def get_client_data(dataset_name):
    """Get clients data based on dataset_name parameter.
    Dataset name is of form: "{folder}/{dataset}_{skew_type}_skew_{skew}_{nr_clients}clients"

    Args:
        dataset_name (str): path to the dataset/experiment

    Returns:
        pandas.DataFrame: dataframe containing the clients experiment data
    """
    if ".csv" not in dataset_name:
        dataset_name = dataset_name + ".csv"

    data = pd.read_csv(dataset_name, sep=",", header="infer")

    # client data discovered by local grid search
    clientData = data[data.clientID > 0]
    clientData = clientData.sort_values(by=["clientID", "accuracy"], ascending=False)

    return clientData


def print_clustering_info(clusterFn, title, data):
    """Print additional information for each cluster in data.
    Additional information includes: size of cluster, number of unique clients,
    min/max/avg/mode/std accuracy per cluster, etc.

    Args:
        clusterFn (sklearn.DBSCAN): cluster function
        title (str): title to print at the beginning
        data (pandas.DataFrame): datapoints that were clustered
    """
    labels = clusterFn.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    clusterIDs = list(range(0, n_clusters_))

    print(f"{title}:")
    print(f"Size of Clients Local Good Params Dataset: {data.shape[0]}")
    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")
    print("--------------")

    for x, y in enumerate(clusterIDs):
        ind_list = list(np.where(labels == y)[0])
        data_cluster = data.iloc[ind_list]

        print(f"Cluster id: {x}")
        print(f"Cluster size: {len(ind_list)}")
        print(f"Unique clients in cluster: {data_cluster.clientID.unique()}")
        print(data_cluster.head())

        print(f"Cluster Min Accuracy: {data_cluster.accuracy.min()}")
        print(f"Cluster Max Accuracy: {data_cluster.accuracy.max()}")
        print(f"Cluster Avg Accuracy: {data_cluster.accuracy.mean()}")
        print(f"Cluster Std Accuracy: {data_cluster.accuracy.std()}")
        print(f"Cluster Median Accuracy: {data_cluster.accuracy.median()}")
        print("--------------")


def clustering(
    dataset_name,
    useDbscan=True,
    metric="euclidean",
    shouldPrint=False,
    drawFig=False,
    saveFig=False,
    top_gt_acc_percentile=0.3,
    top_client_acc_percentile=0.15,
    min_local_accuracy=0.4,
    further_restrict=True,
):
    """Perform clustering on clients local HPs data, plot with ground truth data.

    Args:
        dataset_name (str): path to the experiment file
        useDbscan (bool, optional): whether to use DBSCAN or optics. Defaults to True (DBSCAN).
        metric (str, optional): type of metric to be used in DBSCAN. Defaults to "euclidean".
        shouldPrint (bool, optional): add additional printing. Defaults to False.
        drawFig (bool, optional): whether to plot results. Defaults to False.
        saveFig (bool, optional): whether to save plotted results. If drawFig=False, saveFig has no effect. Defaults to False.
        top_gt_acc_percentile (float, optional): top % of ground truth HPs w.r.t. accuracy to be plotted. Defaults to 0.3.
        top_client_acc_percentile (float, optional): top % of individual clients HPs w.r.t. accuracy to be kept/sent to server. Defaults to 0.15.
        min_local_accuracy (float, optional): minimal local accuracy needed for keeping local HPs. Defaults to 0.4.
        further_restrict (bool, optional): remove all local HPs if all HPs achieve same results, restrict to a maximum number of local HPs. Defaults to True.

    Returns:
        (pandas.DataFrame, pandas.DataFrame, sklearn.DBSCAN, sklearn.DBSCAN): (ground truth data, clients data, cluster function in 2d, cluster function in 3d)
    """
    # ground truth data discovered by federated grid search
    gt = get_gt_data(dataset_name)
    gt = gt[["server_lr", "server_mom", "batch_size", "accuracy"]]

    top_accuracy = gt["accuracy"].max()
    accuracy_threshold = top_accuracy * (
        1 - top_gt_acc_percentile
    )  # This is only used for visualising
    gt = gt[gt["accuracy"] > accuracy_threshold]

    if shouldPrint:
        print("Dataset:", dataset_name)
        print(f"Best accuracy achieved from Ground Truth: {top_accuracy}")
        print(gt.head())
        print(f"# of Optimal Federated Grid Search Params: {gt.shape[0]}")

    # get the data from each client as discovered by their local grid search
    X = get_client_data(dataset_name)
    samples_per_client = X[X["clientID"] == 1].shape[0]
    clientIDs = np.unique(X["clientID"].tolist())

    dataset_short_name = dataset_name.split("/")[-1]
    dataset_info = dataset_short_name.split("_")

    df = pd.DataFrame()
    for client_id in clientIDs:
        tmp = X.loc[X.clientID == client_id]
        top_client_accuracy = tmp["accuracy"].max()
        if top_client_accuracy < min_local_accuracy:
            continue
        client_accuracy_threshold = top_client_accuracy * (
            1 - top_client_acc_percentile
        )  # HP in top top_client_acc_percentile
        tmp = tmp[tmp["accuracy"] > client_accuracy_threshold]
        tmp = tmp[["clientID", "client_lr", "client_mom", "batch_size", "accuracy"]]
        df = pd.concat([df, tmp])

    if further_restrict:
        count_samples = df.groupby("clientID").count()
        df = df[
            ~df.clientID.isin(
                list(count_samples[count_samples.client_lr == samples_per_client].index)
            )
        ]  # If every HP setting is good, we dont care about this client
        df = (
            df.sort_values(by=["accuracy"], ascending=False)
            .groupby("clientID")
            .head(10)
        )

    # Scaling of local params
    scaler = preprocessing.MinMaxScaler().fit(df[["client_lr", "client_mom"]])
    df_scaled = scaler.transform(df[["client_lr", "client_mom"]])

    scaler_3d = preprocessing.MinMaxScaler().fit(
        df[["client_lr", "client_mom", "batch_size"]]
    )
    df_scaled_3d = scaler_3d.transform(df[["client_lr", "client_mom", "batch_size"]])

    # Clustering on the dataset of all the clients
    if useDbscan:
        clusterFn = DBSCAN(eps=0.3, min_samples=2 * 2, metric=metric).fit(df_scaled)
        clusterFn_3d = DBSCAN(eps=0.3, min_samples=2 * 3, metric=metric).fit(
            df_scaled_3d
        )

        if dataset_info[1] == "qty":
            clusterFn = DBSCAN(eps=0.3, min_samples=2, metric=metric).fit(df_scaled)
            clusterFn_3d = DBSCAN(eps=0.3, min_samples=4, metric=metric).fit(
                df_scaled_3d
            )
    else:
        clusterFn = OPTICS(min_samples=2 * 2, metric=metric).fit(df_scaled)
        clusterFn_3d = OPTICS(min_samples=2 * 3, metric=metric).fit(df_scaled_3d)

        if dataset_info[1] == "qty":
            clusterFn = OPTICS(min_samples=2, metric=metric).fit(df_scaled)
            clusterFn_3d = OPTICS(min_samples=4, metric=metric).fit(df_scaled_3d)

    if shouldPrint:
        print_clustering_info(
            clusterFn=clusterFn, title="Clustering info for 2D", data=df
        )
        print_clustering_info(
            clusterFn=clusterFn_3d, title="Clustering info for 3D", data=df
        )

    if drawFig:
        fig = plt.figure(figsize=(12, 24))
        ax_3d = fig.add_subplot(2, 1, 1, projection="3d")
        plt.title(f"{top_accuracy} {dataset_name.split('/')[1]}")
        ax = fig.add_subplot(2, 1, 2)

        plot_GT(ax=ax_3d, gt=gt, is3D=True)
        plot_client_clusters(ax=ax_3d, data=df, clusterFn=clusterFn_3d, is3D=True)

        plot_GT(ax=ax, gt=gt, is3D=False)
        plot_client_clusters(ax=ax, data=df, clusterFn=clusterFn, is3D=False)

        if saveFig:
            plt.savefig(dataset_name + "_DBSCAN" + ".pdf")

    return gt, df, clusterFn, clusterFn_3d


def get_derived_params(ind_list, clientData, dataset_file_name, fedavg_acc):
    """Derive global HPs based on optimal clusters formed with clients data.

    Args:
        ind_list (list): list of indices of optimal clusters
        clientData (pandas.DataFrame): clients datapoints as dataframe
        dataset_file_name (str): path to the experiment file
        fedavg_acc (float): best accuracy achieved by federated grid search

    Returns:
        list: list of dicts with info about derived HPs, closes GT value, difference to best accuracy
    """
    derived_params = list()

    for i, index in enumerate(ind_list):
        d_lr, d_mom, d_bs = average_params(index, clientData)
        # d_lr, d_mom, d_bs = average_params_weighted_acc(index, clientData)

        (c_lr, c_mom, c_bs, c_acc), min_diff = closest_gt_to_point(
            [d_lr, d_mom, d_bs], dataset_file_name
        )
        (c_lrH, c_momH, c_bsH, c_accH), min_diffH = closest_gt_to_point(
            [d_lr, d_mom, d_bs / 2], dataset_file_name
        )

        derived_params.append(
            {
                "num_pts_in_cluster": len(ind_list),
                "lr_mom_bs": (d_lr, d_mom, int(d_bs)),
                "closest_lr_mom_bs_acc": (c_lr, c_mom, c_bs, c_acc),
                "diff_acc": fedavg_acc - c_acc,
                "diff_space": min_diff,
                "heurBS": {
                    "lr_mom_bs_HEURBS": (d_lr, d_mom, int(d_bs) / 2),
                    "closest_lr_mom_bs_acc_HEURBS": (c_lrH, c_momH, c_bsH, c_accH),
                    "diff_acc_HEURBS": fedavg_acc - c_accH,
                    "diff_space_HEURBS": min_diffH,
                },
            }
        )

    return derived_params


def plot_GT(ax, gt, is3D=False):
    """Plot ground truth data in 2d or 3d.

    Args:
        ax (matplotlib.axes.Axes3DSubplot/matplotlib.axes.AxesSubplot): axes to be used for plots
        gt (pandas.DataFrame): ground truth data
        is3D (bool, optional): whether to use 3d or 2d plot. Defaults to False.
    """
    if is3D:
        ax.scatter(
            gt.iloc[:, 0],
            gt.iloc[:, 1],
            gt.iloc[:, 2],
            color="b",
            label="GT",
            marker="x",
            s=250,
        )
        ax.scatter(
            gt.iloc[0, 0],
            gt.iloc[0, 1],
            gt.iloc[0, 2],
            color="b",
            label="GT",
            marker="D",
            s=250,
            alpha=0.6,
        )

        d = defaultdict(dict)
        point_values = list(
            zip(gt.iloc[:, 0], gt.iloc[:, 1], gt.iloc[:, 2], gt.iloc[:, 3])
        )
        point_values.sort(key=lambda x: -x[3])
        for x, y, z, acc in point_values:
            current_acc = d.get((x, y, z), -1)
            if acc > current_acc:
                ax.text(x, y, z, round(acc, 2))
                d[(x, y, z)] = acc
    else:
        ax.scatter(
            gt.iloc[:, 0], gt.iloc[:, 1], color="b", label="GT", marker="x", s=250
        )
        ax.scatter(
            gt.iloc[0, 0],
            gt.iloc[0, 1],
            color="b",
            label="GT",
            marker="D",
            s=250,
            alpha=0.6,
        )

        d = defaultdict(dict)
        point_values = list(zip(gt.iloc[:, 0], gt.iloc[:, 1], gt.iloc[:, 3]))
        point_values.sort(key=lambda x: -x[2])  # sort by accuracy
        for x, y, acc in point_values:
            current_acc = d.get((x, y), -1)
            if acc > current_acc:
                ax.text(x, y, round(acc, 3))
                d[(x, y)] = acc


def plot_client_clusters(ax, data, clusterFn, is3D=False):
    """Plot clients clustered data in 2d or 3d.

    Args:
        ax (matplotlib.axes.Axes3DSubplot/matplotlib.axes.AxesSubplot): axes to be used for plots
        data (pandas.DataFrame): clients data
        clusterFn (sklearn.DBSCAN): cluster function
        is3D (bool, optional): whether to use 3d or 2d plot. Defaults to False.
    """
    labels = clusterFn.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusterIDs = list(range(0, n_clusters_))

    for clusterID in clusterIDs:
        first_point = True
        for z in range(0, len(labels)):
            if labels[z] == clusterID:
                if first_point:
                    if is3D:
                        ax.scatter(
                            data.iloc[z, 1],
                            data.iloc[z, 2],
                            data.iloc[z, 3],
                            color=COLORS[clusterID - 1],
                            marker=MARKERS[clusterID - 1],
                            label="cluster" + str(clusterID),
                            alpha=0.44,
                        )
                    else:
                        ax.scatter(
                            data.iloc[z, 1],
                            data.iloc[z, 2],
                            color=COLORS[clusterID - 1],
                            marker=MARKERS[clusterID - 1],
                            label="cluster" + str(clusterID),
                            alpha=0.44,
                        )
                    first_point = False
                else:
                    if is3D:
                        ax.scatter(
                            data.iloc[z, 1],
                            data.iloc[z, 2],
                            data.iloc[z, 3],
                            color=COLORS[clusterID - 1],
                            marker=MARKERS[clusterID - 1],
                            alpha=0.44,
                        )
                    else:
                        ax.scatter(
                            data.iloc[z, 1],
                            data.iloc[z, 2],
                            color=COLORS[clusterID - 1],
                            marker=MARKERS[clusterID - 1],
                            alpha=0.44,
                        )

    ax.set_xlabel("learning rate")
    ax.set_ylabel("momentum")
    ax.legend()

    if is3D:
        ax.set_zlabel("batch size")


def plot_points_agains_gt_3d(gt, derived):
    """Plot derived data points on top of GT data points in 3d.

    Args:
        gt (pandas.DataFrame): ground truth dataframe
        derived (pandas.DataFrame): derived HPs dataframe
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    plot_GT(ax, gt, is3D=True)
    ax.scatter(
        derived.iloc[:, 0],
        derived.iloc[:, 1],
        derived.iloc[:, 2],
        color="r",
        label="derived",
        marker="o",
        s=250,
    )

    ax.set_xlabel("learning rate")
    ax.set_ylabel("momentum")
    ax.set_zlabel("batch size")
    plt.legend()


def plot_points_agains_gt_2d(gt, derived):
    """Plot derived data points on top of GT data points in 2d.

    Args:
        gt (pandas.DataFrame): ground truth dataframe
        derived (pandas.DataFrame): derived HPs dataframe
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    plot_GT(ax, gt, is3D=False)
    ax.scatter(
        derived.iloc[:, 0],
        derived.iloc[:, 1],
        color="r",
        label="derived",
        marker="o",
        s=250,
    )

    ax.set_xlabel("learning rate")
    ax.set_ylabel("momentum")
    plt.legend()


def closest_gt_to_point(point, dataset_name):
    """Get the closest point from GT points given a point.

    Args:
        point (list): point for which to find the closes GT point
        dataset_name (str): experiment file path

    Returns:
        (list, float): (GT closest point, distance difference to original point)
    """
    gt = get_gt_data(dataset_name)
    gt = gt[["server_lr", "server_mom", "batch_size", "accuracy"]]

    dist = cdist([point], gt.iloc[:, :3])
    idx = dist.argmin()

    return gt.iloc[idx, :4], dist.min()  # returns GT closes to point and distance diff


COLORS = ['deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy', 'royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'r', 'y', 'g', 'k', 'c', 'm', 'violet', 'steelblue', 'lime', 'lightcoral']
MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'P', '*', 'D', 'o', 'v', '^', '<', '>', '8', 's', 'P', '*', 'D']

EXTENSION = ".csv"
DATA_FOLDER = "data"                                      # folder contains csv files for analyzing

DATASETS = glob.glob(DATA_FOLDER + f"/*{EXTENSION}")      # get all csv file names
DATASET_NAMES = [x[:-len(EXTENSION)] for x in DATASETS]   # remove extension from file names
DATASET_NAMES.sort()


# Uncomment for using it as script
# for dataset_name in DATASET_NAMES:
#     clustering(dataset_name, useDbscan=True, shouldPrint=True, drawFig=True, saveFig=True, further_restrict=True)

import numpy as np
import pandas as pd

function_dict_groupby = {
    "mean": pd.core.groupby.GroupBy.mean,
    "max": pd.core.groupby.GroupBy.max,
    "min": pd.core.groupby.GroupBy.min,
    "mode": pd.core.groupby.GroupBy.median,
}
function_dict_series = {
    "mean": pd.Series.mean,
    "max": pd.Series.max,
    "min": pd.Series.min,
    "mode": pd.Series.median,
}


def scoring_function(df, clusterFn, func_name="mean", tolerance=0.0, shouldPrint=False):
    """Ranks and returns optimal clusters based on the scoring function defined with func_name while taking into account the tolerance parameter.

    Args:
        df (pandas.DataFrame): all datapoints
        clusterFn (sklearn.DBSCAN): result of clustering the datapoints in terms of clustering function
        func_name (str, optional): scoring function to be applied to cluster in order to rank them. Defaults to "mean".
        tolerance (float, optional): tolerance when choosing optimal clusters in range 0-1 (e.g. for 5% tolerance choose 0.05). Defaults to 0.0.
        shouldPrint (bool, optional): printing cluster information. Defaults to False.

    Raises:
        Exception: raised when invalid scoring function name is provided. Scoring function can be: mean, max, min, mode.

    Returns:
        list: list of indices of datapoints that belong to optimal cluster, for every optimal cluster
    """

    func_groupby = function_dict_groupby.get(func_name, lambda: "Invalid")
    func_series = function_dict_series.get(func_name, lambda: "Invalid")
    if func_groupby == "Invalid" or func_series == "Invalid":
        raise Exception(
            "Invalid function name provided for scoring function. Scoring function can be: mean, max, min, mode."
        )

    labels = clusterFn.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    clusterIDs = list(range(0, n_clusters_))

    if shouldPrint:
        print("Size of Clients Local Good Params Dataset: %d" % df.shape[0])
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print("--------------")

    clusterData = extend_df_with_clustering_info(df, labels)

    best_acc = func_groupby(clusterData.groupby("cluster"))["accuracy"].max()
    best_list = []  # list of tuples [(ind_list, func_acc)]

    for x, y in enumerate(clusterIDs):
        ind_list = list(np.where(labels == y)[0])
        mode_acc = func_series(df.iloc[ind_list].accuracy)

        if abs(mode_acc - best_acc) < tolerance:
            best_list.append((ind_list, mode_acc))

        if shouldPrint:
            print_cluster_info(x, df.iloc[ind_list])

    best_list_sorted = sorted(best_list, key=lambda x: -x[1])
    best_indxs = [cluster[0] for cluster in best_list_sorted]

    return best_indxs[:4]


def extend_df_with_clustering_info(df, labels, removeNoise=True):
    """Add cluster labels to each data point in df (i.e., introduce new column 'cluster'). Optionaly, remove noise points.

    Args:
        df (pandas.DataFrame): dataframe of points to be labeled
        labels (numpy.ndarray): list of labels
        removeNoise (bool, optional): whether to remove noise points from dataframe. Defaults to True.

    Returns:
        pandas.DataFrame: dataframe after labeling
    """
    clusterData = df.copy(True).reset_index()
    clusterData["cluster"] = -1
    clusterData["cluster"] = clusterData.apply(lambda row: labels[row.name], axis=1)

    if removeNoise:
        clusterData = clusterData[clusterData.cluster != -1]

    return clusterData


def print_cluster_info(cluster_id, clusterPts):
    """Print min/max/avg/mode/std value of the cluster given by clusterPts dataset.

    Args:
        cluster_id (int): ID of the cluster
        clusterPts (pandas.DataFrame): cluster datapoints
    """
    print("Cluster id: %d" % cluster_id)
    print("Cluster size: %d" % clusterPts.shape[0])
    print("Unique clients: ", clusterPts.clientID.unique())

    print("Cluster Min Accuracy: %f" % (clusterPts.accuracy.min()))
    print("Cluster Max Accuracy: %f" % (clusterPts.accuracy.max()))
    print("Cluster Avg Accuracy: %f" % (clusterPts.accuracy.mean()))
    print("Cluster Median Accuracy: %f" % (clusterPts.accuracy.median()))
    print("Cluster Std Accuracy: %f" % (clusterPts.accuracy.std()))
    print("--------------")

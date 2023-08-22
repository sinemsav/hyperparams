"""
Script used for visualizing client HPs vs. ground truth HPs in 2d and 3d.
Preprocessing is applied.

DATA_FOLDER - folder with the csv files to be analyzed
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from clustering import plot_GT

DATA_FOLDER = "data"  # folder contains csv files for analyzing

datasets = glob.glob(DATA_FOLDER + "/*.csv")
dataset_names = [".".join(x.split(".", 2)[:2]) for x in datasets]

colors = ['deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy', 'royalblue', 'maroon', 'forestgreen',
          'mediumorchid', 'tan', 'r', 'y', 'g', 'k', 'c', 'm', 'violet', 'steelblue', 'lime', 'lightcoral']
markers = ['o', 'v', '^', '<', '>', '8', 's', 'P', '*', 'D', 'o', 'v', '^', '<', '>', '8', 's', 'P', '*', 'D']
sizes = [360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20, 10, 5]

for dataset_name in dataset_names:
    print("Dataset / Setting:", dataset_name)
    data = pd.read_csv(dataset_name + ".csv", sep=",", header="infer")

    # ground truth data discovered by federated grid search
    gt = data[(data.clientID == -1)]
    gt = gt.sort_values(by="accuracy", ascending=False)
    gt = gt[["server_lr", "server_mom", "batch_size", "accuracy"]]

    # find the max accuracy and set a threshold
    top_accuracy = gt["accuracy"].max()
    print("Best accuracy: %f" % top_accuracy)
    accuracy_threshold = top_accuracy - top_accuracy * 0.3  # top 30%
    gt = gt[gt["accuracy"] > accuracy_threshold]

    print("# of Ground Truth Params:", gt.shape[0])

    # each client data discovered by local grid search
    X = data[data.clientID > 0]
    X = X.sort_values(by=["clientID", "accuracy"], ascending=False)
    samples_per_client = X[X["clientID"] == 1].shape[0]
    clientIDs = np.unique(X["clientID"].tolist())

    df = pd.DataFrame()
    for client_id in clientIDs:
        tmp = X.loc[data.clientID == client_id]
        top_client_accuracy = tmp["accuracy"].max()
        if top_client_accuracy < 0.4:  # remove HPs below 40% acc
            continue
        client_accuracy_threshold = (
            top_client_accuracy - top_client_accuracy * 0.05
        )  # top 5%
        tmp = tmp[tmp["accuracy"] > client_accuracy_threshold]
        tmp = tmp[["clientID", "client_lr", "client_mom", "batch_size", "accuracy"]]
        df = pd.concat([df, tmp])

    count_samples = df.groupby("clientID").count()
    df = df[
        ~df.clientID.isin(
            list(count_samples[count_samples.client_lr == samples_per_client].index)
        )
    ]
    df = df.sort_values(by=["accuracy"], ascending=False).groupby("clientID").head(10)

    clientIDs = df["clientID"].tolist()

    fig = plt.figure(figsize=(12, 24))
    ax3d = fig.add_subplot(2, 1, 1, projection="3d")
    plt.title(f"{top_accuracy} {dataset_name.split('/')[1]}")
    ax = fig.add_subplot(2, 1, 2)

    # plot client data in 2d and 3d
    for x, y in enumerate(clientIDs):
        if x % df[df.clientID == y].shape[0] == 0:
            ax3d.scatter(
                df.iloc[x, 1],
                df.iloc[x, 2],
                df.iloc[x, 3],
                color=colors[y - 1],
                marker=markers[y - 1],
                label="client" + str(y),
                s=sizes[y - 1],
                alpha=0.44,
            )
            ax.scatter(
                df.iloc[x, 1],
                df.iloc[x, 2],
                color=colors[y - 1],
                marker=markers[y - 1],
                label="client" + str(y),
                s=sizes[y - 1],
                alpha=0.44,
            )
        else:
            ax3d.scatter(
                df.iloc[x, 1],
                df.iloc[x, 2],
                df.iloc[x, 3],
                color=colors[y - 1],
                marker=markers[y - 1],
                s=sizes[y - 1],
                alpha=0.44,
            )
            ax.scatter(
                df.iloc[x, 1],
                df.iloc[x, 2],
                color=colors[y - 1],
                marker=markers[y - 1],
                s=sizes[y - 1],
                alpha=0.44,
            )

    plot_GT(ax=ax3d, gt=gt, is3D=True)
    plot_GT(ax=ax, gt=gt, is3D=False)

    ax.set_xlabel("learning rate")
    ax.set_ylabel("momentum")

    ax3d.set_xlabel("learning rate")
    ax3d.set_ylabel("momentum")
    ax3d.set_zlabel("batch size")

    ax.legend()
    ax3d.legend()

    plt.savefig(dataset_name + "_" + "visual" + ".pdf")

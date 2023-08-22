from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import collections
import numpy as np


def display_dataset_barplot(labels, num_classes, title=""):
    """Display label distribution of the dataset

    :param labels: training labels of the dataset
    :param num_classes: number of classes in the dataset
    :param title: title of the barplot (default is empty string "")
    """

    f = plt.figure(figsize=(12, 7))
    plot_data = collections.defaultdict(list)

    for l in labels:
        plot_data[l].append(l)
    plt.subplot(1, 1, 1)
    plt.title(title)
    for j in range(num_classes):
        plt.hist(
            plot_data[j],
            density=False,
            bins=np.arange(num_classes + 1))
    plt.show()


def display_per_client_barplot(clientsDataLabels, num_clients, num_classes, decentralized):
    """Display label distribution per client

    :param clientsDataLabels: labels for each of the clients
    :param num_clients: number of clients
    :param num_classes: number of classes in the dataset
    :param decentralized: True if running decentralized experiment
    """

    f = plt.figure(figsize=(12, 12))
    for i in range(num_clients):

        # build subplot for each client

        client_labels = clientsDataLabels[i]
        plot_data = collections.defaultdict(list)

        if not decentralized:
            for l in client_labels:
                plot_data[l].append(l)
        else:
            for l in client_labels:
                plot_data[np.argmax(l)].append(np.argmax(l))

        plt.subplot(num_clients, 1, i + 1)

        plt.title('Client {}'.format(i))

        for j in range(num_classes):
            plt.hist(
                plot_data[j],
                density=False,
                bins=np.arange(num_classes + 1))
    plt.show()


def display_heatmap(dirimap, labels, num_clients, num_classes, decentralized):
    """Display heatmap of the label distribution amongst the clients

    :param dirimap: map containing the distribution of samples IDs for each client
    :param labels: dataset labels
    :param num_clients: number of clients
    :param num_classes: number of classes
    :param decentralized: True if running decentralized experiment
    """
    heatmap = []

    for k in dirimap:
        # for each client, count the number of each class given by the dirimap
        client_label_distrib = np.zeros(num_classes, dtype=int)

        if not decentralized:
            for e in dirimap[k]:
                client_label_distrib[labels[e]] += 1
        else:
            for e in dirimap[k]:
                client_label_distrib[np.argmax(labels[e])] += 1

        heatmap.append(client_label_distrib)

    heatmap = np.array(heatmap).T

    party_ids = np.arange(num_clients)
    labels = np.arange(num_classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("label/client", rotation=-90, va="bottom")

    ax.set_xticks(party_ids)
    ax.set_yticks(labels)

    ax.set_xticklabels(party_ids)
    ax.set_yticklabels(labels)

    ax.set_xlabel('Clients')
    ax.set_ylabel('Labels')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_clients):
            text = ax.text(j, i, heatmap[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Distribution of labels amongst clients")
    fig.tight_layout()
    plt.show()

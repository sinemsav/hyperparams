import collections
import tensorflow_federated as tff
import numpy as np
from matplotlib import pyplot as plt
from federated_library.display_distribution import display_heatmap,\
    display_per_client_barplot
from fedjax import InMemoryFederatedData


def to_ClientData(
    clientsData: np.ndarray, clientsDataLabels: np.ndarray,
    ds_info, is_train=True
) -> tff.simulation.datasets.TestClientData:
    """Transform dataset to be fed to fedjax

    :param clientsData: dataset for each client
    :param clientsDataLabels:
    :param ds_info: dataset information
    :param train: True if processing train split
    :return: dataset for each client cast into TestClientData
    """
    num_clients = ds_info['num_clients']

    client_data = collections.OrderedDict()

    for i in range(num_clients if is_train else 1):
        client_data[str(i)] = collections.OrderedDict(
            x=clientsData[i],
            y=clientsDataLabels[i])

    return tff.simulation.datasets.TestClientData(client_data)


def convert_to_federated_data(
    clientsData: np.ndarray, clientsDataLabels: np.ndarray,
    ds_info, is_train=True
) -> InMemoryFederatedData:
    """Transform dataset to be fed to fedjax

    :param clientsData: dataset for each client
    :param clientsDataLabels:
    :param ds_info: dataset information
    :param train: True if processing train split
    :return: dataset for each client cast into InMemoryFederatedData
    """
    num_clients = ds_info['num_clients']

    client_data = collections.OrderedDict()

    if is_train:
        for i in range(num_clients):
            # print(f"Client {i}: {clientsData[i].shape} array.")
            client_data[str(i)] = {
                "x": clientsData[i],
                "y": clientsDataLabels[i]
            }
    else:
        # print(f"Test data: {clientsData.shape} array.")
        client_data["0"] = {
            "x": clientsData,
            "y": clientsDataLabels
        }

    return InMemoryFederatedData(client_data)


def build_ClientData_from_dataidx_map(
        x_train: np.ndarray, y_train: np.ndarray,
        ds_info, dataidx_map, decentralized, display, is_tf):
    """Build dataset for each client based on  dataidx_map

    :param x_train: training split samples
    :param y_train: training split labels
    :param ds_info: dataset information
    :param dataidx_map: distribution of the labels for each client
    :param decentralized: True if running decentralized experiment
    :param display: True if barplot and heatmap are wanted
    :return: dataset for each client based on dataidx_map
    """

    num_clients = ds_info['num_clients']
    num_classes = ds_info['num_classes']
    sample_shape = ds_info['sample_shape']

    numSamplesPerClient = [len(x) for x in dataidx_map.values()]

    sample_height, sample_width, sample_channels = sample_shape

    clientsData = np.zeros(num_clients, dtype=object)
    clientsDataLabels = np.zeros(num_clients, dtype=object)

    for i in range(num_clients):
        clientData = np.zeros(
            (numSamplesPerClient[i], sample_height, sample_width, sample_channels))

        if decentralized:
            clientDataLabels = np.zeros((numSamplesPerClient[i], num_classes))
        else:
            clientDataLabels = np.zeros((numSamplesPerClient[i]))

        for j, s in enumerate(dataidx_map[i]):
            clientData[j] = x_train[s]
            clientDataLabels[j] = y_train[s]

        shuffler = np.random.permutation(len(clientData))
        clientData = clientData[shuffler]
        clientDataLabels = clientDataLabels[shuffler]

        clientsData[i] = clientData
        clientsDataLabels[i] = clientDataLabels

    if display:
        display_heatmap(dataidx_map, num_classes=num_classes,
                        num_clients=num_clients, labels=y_train,
                        decentralized=decentralized)
        display_per_client_barplot(clientsDataLabels, num_classes=num_classes,
                                   num_clients=num_clients, decentralized=decentralized)

    if decentralized:
        return clientsData, clientsDataLabels
    # If federated setting
    if is_tf:
        return to_ClientData(clientsData, clientsDataLabels, ds_info)
    else:
        return convert_to_federated_data(clientsData, clientsDataLabels, ds_info)


def iid_distrib(x_train: np.ndarray, y_train: np.ndarray, ds_info,
                decentralized, display=False, is_tf=False):
    """Build an iid distributed dataset for each client

    :param x_train: training split samples
    :param y_train: training split labels
    :param ds_info: dataset information
    :param decentralized: True if running decentralized experiment
    :param display: True if barplot is wanted
    :return: dataset for each client with labels iid distributed
    """

    num_clients = ds_info['num_clients']
    num_classes = ds_info['num_classes']
    sample_shape = ds_info['sample_shape']

    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]

    numSamplesPerClient = int(x_train.shape[0] / num_clients)

    sample_height, sample_width, sample_channels = sample_shape

    clientsData = np.zeros((num_clients, int(
        numSamplesPerClient), sample_height, sample_width, sample_channels))

    if decentralized:
        clientsDataLabels = np.zeros(
            (num_clients, int(numSamplesPerClient), num_classes))
    else:
        clientsDataLabels = np.zeros((num_clients, int(numSamplesPerClient)))

    ind = 0
    for i in range(num_clients):
        clientsData[i] = x_train[ind:ind + numSamplesPerClient]
        clientsDataLabels[i] = y_train[ind:ind + numSamplesPerClient]
        ind = ind + numSamplesPerClient

    if display:
        display_per_client_barplot(clientsDataLabels, num_classes=num_classes,
                                   num_clients=num_clients, decentralized=decentralized)

    if decentralized:
        return clientsData, clientsDataLabels
    else:
        if is_tf:
            return to_ClientData(clientsData, clientsDataLabels, ds_info)
        else:
            return convert_to_federated_data(clientsData, clientsDataLabels, ds_info)


def qty_skew_distrib(
        x_train: np.ndarray, y_train: np.ndarray, ds_info, beta,
        decentralized, display=False, is_tf=False):
    """Build an quantity-skewed distributed dataset for each client, with Dirichlet distribution
        of parameter beta

    :param x_train: training split samples
    :param y_train: training split labels
    :param ds_info: dataset information
    :param beta: parameter for Dirichlet distribution, greater beta is more balanced dataset
    :param decentralized: True if running decentralized experiment
    :param display: True if barplot and heatmap are wanted
    :return: dataset for each client with a Dir(beta) quantity-skewed distribution
    """

    num_clients = ds_info['num_clients']
    num_classes = ds_info['num_classes']

    idxs = np.random.permutation(y_train.shape[0])
    min_size = 0
    while min_size < num_classes:
        proportions = np.random.dirichlet(np.repeat(beta, num_clients))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * len(idxs))
    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    batch_idxs = np.split(idxs, proportions)
    net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}

    return build_ClientData_from_dataidx_map(
        x_train, y_train, ds_info, net_dataidx_map,
        decentralized=decentralized, display=display, is_tf=is_tf)


def label_skew_distrib(
        x_train: np.ndarray, y_train: np.ndarray, ds_info, beta,
        decentralized, display=False, is_tf=False):
    """Build an label-skewed distributed dataset for each client, with Dirichlet distribution
        of parameter beta

    :param x_train: training split samples
    :param y_train: training split labels
    :param ds_info: dataset information
    :param beta: parameter for Dirichlet distribution, greater beta is more balanced dataset
    :param decentralized: True if running decentralized experiment
    :param display: True if barplot and heatmap are wanted
    :return: dataset for each client with a Dir(beta) label-skewed distribution
    """

    # https://github.com/Xtra-Computing/NIID-Bench/blob/7a96525cc52dca5bae13266398c123f08b7f833b/utils.py

    num_clients = ds_info['num_clients']
    num_classes = ds_info['num_classes']
    seed = ds_info['seed']

    min_size = 0
    min_require_size = num_classes
    K = num_classes

    if not decentralized:
        y_train_uncat = y_train
    else:
        y_train_uncat = np.array([np.argmax(label_cat)
                                 for label_cat in y_train])

    N = y_train_uncat.shape[0]
    np.random.seed(seed)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            # get indices for class k
            idx_k = np.where(y_train_uncat == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))

            # Balance
            proportions = np.array([p * (len(idx_j) < N / num_clients)
                                   for p, idx_j in zip(proportions, idx_batch)])

            proportions = proportions / proportions.sum()

            proportions = (np.cumsum(proportions) *
                           len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j,
                         idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return build_ClientData_from_dataidx_map(
        x_train, y_train, ds_info, net_dataidx_map,
        decentralized=decentralized, display=display, is_tf=is_tf
    )


def feature_skew_distrib(
        x_train: np.ndarray, y_train: np.ndarray, ds_info, sigma,
        decentralized, display=False, is_tf=False):
    """Build an feature-skewed distributed dataset for each client, adds a Gaussian noise of parameters N(0,
    (sigma * i/num_clients)**2) for each client i

    :param x_train: training split samples
    :param y_train: training split labels
    :param ds_info: dataset information
    :param sigma: standard deviation for the Gaussian distribution
    :param decentralized: True if running decentralized experiment
    :param display: True if image samples are wanted
    :return: dataset for each client with a Dir(beta) feature-skewed distribution
    """

    num_clients = ds_info['num_clients']
    sample_shape = ds_info['sample_shape']

    sample_height, sample_width, sample_channels = sample_shape

    def noise(sigma_i):
        return np.random.normal(
            scale=sigma_i, size=(sample_height, sample_width, sample_channels)
        )

    if not decentralized:
        dataset_iid = iid_distrib(
            x_train, y_train, ds_info, decentralized=decentralized,
            display=display, is_tf=True
        )

        clientsData = np.zeros(num_clients, dtype=object)
        clientsDataLabels = np.zeros(num_clients, dtype=object)

        for i in range(num_clients):
            client_ds = dataset_iid.create_tf_dataset_for_client(
                dataset_iid.client_ids[i])
            client_ds = list(client_ds.as_numpy_iterator())

            client_ds_len = len(client_ds)

            clientData = np.zeros(
                (client_ds_len, sample_height, sample_width, sample_channels))

            clientDataLabels = np.zeros(client_ds_len)

            for j, sample in enumerate(client_ds):
                # Add gaussian noise to the sample
                clientData[j] = np.maximum([0.0], np.minimum(
                    [1.0], sample['x'] + noise(sigma * float(i) / float(num_clients-1))))
                clientDataLabels[j] = sample['y']

                if display and j == 0:
                    plt.imshow(sample['x'].reshape(
                        sample_height, sample_width, sample_channels))
                    plt.show()
                    plt.imshow(clientData[j].reshape(
                        sample_height, sample_width, sample_channels))
                    plt.show()

            clientsData[i] = clientData
            clientsDataLabels[i] = clientDataLabels

        if is_tf:
            return to_ClientData(clientsData, clientsDataLabels, ds_info)
        else:
            return convert_to_federated_data(clientsData, clientsDataLabels, ds_info)
    else:
        clientsData, clientsDataLabels = iid_distrib(
            x_train, y_train, ds_info, decentralized=decentralized, display=display)

        for i in range(num_clients):

            client_ds = clientsData[i]

            client_ds_len = len(client_ds)

            clientData = np.zeros(
                (client_ds_len, sample_height, sample_width, sample_channels))

            for j, sample in enumerate(client_ds):
                # Add gaussian noise to the sample
                clientData[j] = np.maximum([0.0], np.minimum(
                    [1.0], sample + noise(sigma * float(i) / float(num_clients-1))))

                if display and j == 0:
                    plt.imshow(sample.reshape(sample_height,
                               sample_width, sample_channels))
                    plt.show()
                    plt.imshow(clientData[j].reshape(
                        sample_height, sample_width, sample_channels))
                    plt.show()

            clientsData[i] = clientData

        return clientsData, clientsDataLabels

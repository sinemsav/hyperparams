# Grid values for federated experiments

DATASETS = ["mnist", "svhn_cropped", "cifar10", "emnist"]

SKEWS = {
    "feature": [0.02, 0.1],
    "label": [0.1, 1.0, 5.0],
    "qty": [0.1, 0.4, 1.0, 2.0]
}

HP_GRID = {
    "lr": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
    "mom": [0.0, 0.3, 0.6, 0.9],
    "bs": [8, 16, 32],
    "interval": [3, 5, 7, 10, 12]
}

NR_PARTIES = [10, 20]

HEUR_DICT = {
    "lr": {
        "in": "client_lr",
        "out": "server_lr",
    },
    "mom": {
        "in": "client_momentum",
        "out": "server_momentum",
    },
    "bs": {
        "in": "batch_size",
        "out": "batch_size",
    }
}

INPUT_HEUR = [HEUR_DICT[hp]["in"] for hp in HEUR_DICT.keys()]
OUTPUT_HEUR = [HEUR_DICT[hp]["out"] for hp in HEUR_DICT.keys()]

from itertools import product
from pprint import pprint
from copy import deepcopy
import time
from datetime import datetime
import numpy as np
import fedjax
from fedjax.algorithms import fed_avg
import jax
import jax.numpy as jnp

from federated_library.models_haiku import get_model
from federated_library.distributions import qty_skew_distrib, \
    label_skew_distrib, feature_skew_distrib, iid_distrib


def train_fed_avg(params, ds, test_split, ds_info, custom_model=None,
                  display=False):
    """ Run training and test with FedJAX FedAvg

    :param params: parameters to train on, dictionary with keys [skew,
        client_lr, server_lr, client_momentum, server_momentum, batch_size,
        epochs_per_round, rounds]
    :param ds: train dataset
    :param test_split: test dataset
    :param ds_info: dataset information
    :param custom_model: optional custom model
    :param display: True if distributions graphs are wanted
    :return: test metrics result
    """
    skew_type = ds_info['skew_type']
    seed = ds_info['seed']

    x_train = ds[0]
    y_train = ds[1]

    if skew_type == "qty":
        # print("Quantity skew")
        federated_data = qty_skew_distrib(
            x_train, y_train, ds_info, params['skew'], decentralized=False,
            display=display)
    elif skew_type == "label":
        # print("Label skew")
        federated_data = label_skew_distrib(
            x_train, y_train, ds_info, params['skew'], decentralized=False,
            display=display)
    elif skew_type == "feature":
        # print("Feature skew")
        federated_data = feature_skew_distrib(
            x_train, y_train, ds_info, params['skew'], decentralized=False,
            display=display)
    else:
        # print("IID distribution")
        federated_data = iid_distrib(
            x_train, y_train, ds_info, decentralized=False, display=display)

    model = get_model(params, ds_info, custom_model)

    # Scalar loss function with model parameters, batch of examples, and seed
    # PRNGKey as input.
    def loss(params, batch, rng):
        # `rng` used with `apply_for_train` to apply dropout during training.
        preds = model.apply_for_train(params, batch, rng)
        # Per example loss of shape [batch_size].
        example_loss = model.train_loss(batch, preds)
        return jnp.mean(example_loss)

    # Gradient function of `loss` w.r.t. to model `params` (jitted for speed).
    grad_fn = jax.jit(jax.grad(loss))

    # Create federated averaging algorithm.
    client_optimizer = fedjax.optimizers.sgd(learning_rate=params['client_lr'],
                                             momentum=params['client_momentum'])
    server_optimizer = fedjax.optimizers.sgd(learning_rate=params['server_lr'],
                                             momentum=params["server_momentum"])

    # Hyperparameters for client local training dataset preparation.
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
        batch_size=params["batch_size"],
        num_epochs=params['epochs_per_round'])

    algorithm = fed_avg.federated_averaging(
        grad_fn,
        client_optimizer,
        server_optimizer,
        client_batch_hparams
    )

    # Initialize model parameters and algorithm server state
    init_params = model.init(jax.random.PRNGKey(seed))
    server_state = algorithm.init(init_params)
    clients = []

    for client_id, client_dataset in federated_data.clients():
        clients.append((client_id, client_dataset, jax.random.PRNGKey(seed)))

    # Start training
    for round_num in range(params['rounds']):
        server_state, client_diagnostics = algorithm.apply(
            server_state, clients
        )
        # Print client diagnostics if curious about each client's model
        if display:
            print(f"[round {round_num}]")
            pprint(client_diagnostics)

    # Final evaluation
    test_eval_dataset = [cds for _, cds in test_split.get_clients('0')]
    test_eval_batches = fedjax.padded_batch_client_datasets(
        test_eval_dataset, batch_size=256)

    test_metrics = fedjax.evaluate_model(model,
                                         params=server_state.params,
                                         batches=test_eval_batches)

    return test_metrics


def grid_parameters(parameters):
    grid = []
    for params in product(*parameters.values()):
        grid.append(dict(zip(parameters.keys(), params)))

    return grid


def fed_avg_gridsearch(params, ds, test_split, ds_info, display):
    hp_grid = grid_parameters(params)
    print(
        f"Gridsearch on {len(hp_grid)} configurations. "
        f"({datetime.now().strftime('%d/%m/%Y %H:%M:%S')})")

    results = []
    for i, hp_config_params in enumerate(hp_grid):
        tic = time.time()

        run_res_list = []
        # Take mean accuracy of runs
        for _ in range(hp_config_params["runs"]):
            run_res = train_fed_avg(
                hp_config_params, ds, test_split, ds_info,
                display=display)['accuracy']

            run_res_list.append(run_res)

        mean_run_res = float(jnp.mean(jnp.array(run_res_list)))
        hp_config_formatted = deepcopy(hp_config_params)
        hp_config_formatted["act_fn"] = hp_config_formatted["act_fn"].__name__
        results.append((mean_run_res, hp_config_formatted))

        toc = time.time()
        print(
            f"{i + 1}. configuration finished (Accuracy: {mean_run_res}, "
            f"Time: {(toc - tic):2f} s)")

    return results


def fed_avg_intervalsearch(params, ds, test_split, ds_info, display):
    results = []

    for interval in params['intervals']:

        acc_runs = np.zeros(params['runs'], dtype=object)

        params['interval'] = interval

        print(f'Training with params: {params}')

        for r in range(params['runs']):
            acc_runs[r] = train_fed_avg(
                params, ds, test_split, ds_info, display=display)['accuracy']

        avg_acc_runs = np.mean(acc_runs)

        results.append((avg_acc_runs, deepcopy(params)))

    return results

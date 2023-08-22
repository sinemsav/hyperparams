# Non-IID Scripts

The Non-IID scripts contain experiments in a federated setting (with clients collaboratively training a joint model).

## Cluster setup

Local experiments are those computed by each client independently. Federated experiments are those computed using Federated Averaging. The experiments are assumed to be conducted on a cluster, connected to the user device via SSH.

Check out `ic_cluster_setup.md` for details on setting up the IC cluster running Ubuntu 20.04 from scratch.

## Python environments

The experiments (local and federated) use different Python environments. Both can be run either on CPU or GPU.

To create the necessary Python virtual envirnments, make sure `python3-venv` is installed.

Side note: If you want to kill all running jupyter notebooks, run: `pkill jupyter`

### Local

Check Python version with `python3 -V`. It must be version greater than or equal to 3.6.X.

Run the script to install the local virtual environment on the server: `./local_experiments/install_local_env.sh`. It will install the environment and launch `jupyter notebook` in the background on port 8890.

To connect with the notebook from your device, open an SSH bridge by running:
`ssh -N -f -L localhost:{LOCAL_PORT}:localhost:8890 {server_user}@{server_address}`.
Then go to `localhost:{LOCAL_PORT}` on your browser.

Requirements installed by the script: `pip, wheel, jupyterlab, ipywidgets, numpy, talos, pandas, matplotlib, tensorflow_datasets, tensorflow_federated==0.18.0`

### Federated

#### CPU support

Run the script to install the federated virtual environment on the server: `./federated_experiments/install_fed_env.sh cpu`. It will install the environment and launch `jupyter notebook` in the background on port 8891.

Requirements installed by the script: `pip, talos, wheel, jupyterlab, ipywidgets, numpy, pandas, matplotlib, tensorflow_datasets, tensorflow_federated, jax, fedjax`

To connect with the notebook on your own computer, open an SSH bridge by running:
`ssh -N -f -L localhost:{LOCAL_PORT}:localhost:8891 {server_user}@{server_address}`.
Then go to `localhost:{LOCAL_PORT}` on your browser.

#### GPU support

For GPU support, CUDA has to be installed:

- CUDA:
  - The script assumes CUDA version 11.
  - Check the CUDA version installed with `nvcc --version`.
  - The `nvcc` version should correspond to the version displayed with the command `nvidia-smi`
  - To check all installed CUDA versions, run:

    ```{bash}
    function lib_installed() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
    function check() { lib_installed $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }
    check libcuda
    check libcudart
    ```

- cuDNN:
  - The script assumes cuDNN version 8.0.5
  - To check the cuDNN version installed, run:

    ```{bash}
    function lib_installed() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
    function check() { lib_installed $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }
    check libcudnn
    ```

For other versions, change the `jax` version in the install script in accordance with [this document](https://github.com/google/jax/blob/main/README.md#pip-installation-gpu-cuda).

Run the following script to install the federated virtual environment on the server: `./federated_experiments/install_fed_env.sh gpu`. It will install the virtual environment and launch `jupyter notebook` in the background on port 8889.

Requirements installed by the script: `pip, talos, wheel, jupyterlab, ipywidgets, numpy, pandas, matplotlib, tensorflow_datasets, tensorflow_federated, jax, fedjax`

To connect with the notebook from your device, create an SSH bridge running:
`ssh -N -f -L localhost:{LOCAL_PORT}:localhost:8889 {server_user}@{server_address}`
Then go to `localhost:{LOCAL_PORT}` on your browser.

## Python files

## `constants.py`

File containing constants used throughout the experiments. It contains datasets, types and levels of skew, as well as the hyperparameter grids. It serves as ground truth for experiments.

## Grid Search Notebooks

Possible `skew_type`s are `qty`, `label`, and `feature`.

Possible datasets can be found in [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/overview).

Our experiments make use of [`mnist`]([www.google.com](https://www.tensorflow.org/datasets/catalog/mnist)), [`emnist`](https://www.tensorflow.org/datasets/catalog/emnist), [`cifar10`](https://www.tensorflow.org/datasets/catalog/cifar10), and [`svhn_cropped`](https://www.tensorflow.org/datasets/catalog/svhn_cropped).

### Local Experiments

#### `local_non-iid.ipynb` notebook

What: individually perform grid search and interval search on each client (in a local setting)

How:

1. Load dataset and generate test split
    - call `load_tf_dataset`, then transform the `ds_test` with `tf.data.Dataset.from_tensor_slices` and generate batches with the `batch(n)` function.

2. call the run function with the following arguments:
    - `hyperparams`: dict of lists whose keys are `act_fn`, `act_fn_approx` (not mandatory if `with_intervals` is set to `False`), `intervals` (not mandatory if `with_intervals` is set to `False`), `client_lr`, `client_momentum`, `batch_size`, `epochs`, `clients_set`, `skews_set`
    - `ds`: dataset from `load_tf_dataset`
    - `test_dataset`: transformed and batched test dataset from 1.
    - `ds_info`: additional dataset information
    - `with_intervals`: True if you want to perform interval search
    - `display`: True if you want to see the visual representation of the distributions

The script will create a folder named `{dataset_name}_non_iid_res` . It will be used to receive the results of the gridsearch. 3 types of files (or 2 if `with_intervals` is set to `False` ) will be generated:

- text file with "distribution" keyword: number of samples and ratio per client for quantity skew, number of samples of each class per client for label skew.
- text file with "intervals" keyword: each line corresponds to a client and its best interval.
- other text files: result of the grid search, each line is a tuple (validation_accuracy, hyperparameters used), the file is sorted in decreasing order by validation_accuracy.

You can also tune the callbacks (i.e. early stopping) in the `experiment` function.

### Federated Experiments

#### `fedJAX_gridSearch.ipynb` notebook

What: perform federated grid search

How:

1. Load dataset and generate test split
    - call `load_tf_dataset`, then transform the test data `(x_test, y_test)` with `convert_to_federated_data(x_test, y_test, ds_info, is_train=False)`.

2. You can then
    - call the `run` function with the following arguments to run grid search:
    - params: dict of hyperparameter configurations whose keys are `act_fn`, `client_lr`, `server_lr`, `client_momentum`, `server_momentum`, `batch_size`, `epochs_per_round`, `rounds`, `runs` (int, not list), `clients_set`, `skews_set`
    - ds: dataset from `load_tf_dataset`
    - test_split: test dataset from 1.
    - ds_info: dataset information
    - display: True if you want to display the visual representation of the distributions.

The script will return the result of the grid search performed by the `run` function in a text file with the format `{dataset_name}_{skew_type}_{skew}_{parties}clients.txt`. Each file contains tuples of (test_accuracy, hyperparameter configuration), sorted in descending order by test_accuracy.

#### `fedJAX_intervalSearch.ipynb` notebook

What: perform interval search with in a federated learning setting

How:

1. Load dataset and generate test split
    - call `load_tf_dataset`, then transform the test data `(x_test, y_test)` with `convert_to_federated_data(x_test, y_test, ds_info, is_train=False)`.

2. Call the `run` function to perform interval search on the given sets of parameters with the following arguments:
    - params: dict of hyperparameter configurations whose keys are `act_fn`, `intervals`, `client_lr`, `server_lr`, `client_momentum`, `server_momentum`, `batch_size`, `epochs_per_round`, `rounds`, `runs` (int, not list), `clients_set`, `skews_set`. Keys `server_lr`, `server_momentum`, `batch_size`, `clients_set` and `skews_set` must be lists such that entry `i` of each list is a set of parameters. You can change this behavior at the beginning of the `run` function to add or remove tunable parameters.
    - ds: dataset from `load_tf_dataset`
    - test_split: test dataset from 1.
    - ds_info: dataset information
    - display: True if you want to display the visual representation of the distributions.

## Results aggregation

## Federated grid search results

In the `fed_grid_search_results/` folder, find the results of the federated grid search experiments for multiple distribution skews and datasets. The hyperparameter grid contains multiple server hyperparameter configurations. In every file, hyperparameter configurations are sorted decreasingly with respect to validation accuracy.

## Non-IID results

In the `non-IID_res/` folder, find the results of:

- `fedavg` (deprecated): federated grid search with grid containing multiple client hyperparameters configurations, and server hyperparameters kept fixed. The results of these experiments are deprecated.
- `individual`: local grid search, performed on every client independently.
- `intervals`: federated grid search with different values of activation function approximation intervals.

### Results aggregation notebooks

Since these notebooks use the `fedavg` results, the notebooks are deprecated.

Requirements: `numpy`, `pandas`

Choose the `DATASETS`, `SKEWS` and `NR_PARTIES` in `constants.py` and run the notebook to perform the local results aggregation and get parameters for FedAvg following the method defined in the `get_res` function. You can change the aggregation method at will to try to get better or worse results with FedAvg.

## Heuristic

For more details on the development of the heuristic functions, check out the README file inside the `heuristic/` folder.

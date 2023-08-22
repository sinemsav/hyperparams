# Heuristic Functions in Non-IID Setting

This folder contains the code related to the development of heuristic functions with the goal to find good global hyperparameters from local computations.

## Python files

## `heuristic_funcs.py`

Description: File contains the developed heuristic functions.

Functions:

- `lr_heuristic`: Learning rate heuristic
- `mom_heuristic`: Momentum heuristic
- `bs_heuristic`: Batch size heuristic
- `aggregate_results`: Function performing heuristic functions for a single (if hp_name not None) or all hyperparameters.

## `display.py`

Description: Functions used to display heuristic experiment results.

Functions:

- `get_heuristic_errors`: Function returning difference between hyperparameters from federated grid search and the heuristic functions and baseline functions. If plot is True, display boxplots of the heuristic function outputs.
- `print_heur_results`: Display output of baselines and heuristic functions compared to the optimal grid search hyperparameters.

## `utils.py`

Description: Collection of Python functions which make common list and dict operations.

## `read_data.py`

Description:

Functions:

- `get_local_res`: Read optimal client hyperparameters from local experiments.
- `get_federated_res`: Read optimal hyperparameter configuration of federated grid search on particular distribution skew setting.
- `get_federated_val_acc`: Read best validation accuracy of federated grid search experiment for specific setting.

## Experiments

### `visualize_heuristics.ipynb`

Description: Visualize output of heuristic functions compared to optimal federated hyperparameters. Print mean absolute error for every heuristic and baseline function. Notebook is used to develop a heuristic function by minimizing the mean absolute error between output and the empirically found optimal global hyperparameter.

### `evaluate_heuristics.ipynb`

Description: Run evaluation of developed heuristic functions. We evaluate the performance of the constructed heuristic functions by performing federated learning using the returned global hyperparameters under multiple settings. We then compare the validation accuracy of the global hyperparameters found by the heuristic functions to the accuracy of the ones found by federated grid search. This notebook is expected to run using the federated Python environment installed in `non_IID_setting/federated_experiments/install_fed_env.sh`.

### `aggregate_heur_results.ipynb`

Description: Generate box plots of evaluation results and compute decrease in validation accuracy

## Results

The `heur_results/` folder contains the results of the evaluated heuristic functions. There is one JSON file for every dataset. The key of the dictionary is the version of the heuristic function, which contains a list of hyperparameter settings with federated and heuristic function validation accuracies.

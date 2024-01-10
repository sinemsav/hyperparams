# DBSCAN-based FL-HPO in Non-IID Setting

This folder contains the code for DBSCAN-based FL-HPO: finding good global hyperparameters by leveraging local HPOs. Moreover, DBSCAN-based FL-HPO can also be performed under encryption which is demonstrated in the [`clustering_federated_dbscan`](./clustering_federated_dbscan) directory.

## Experiments
The experiments used during the project can be found in [`experiments.zip`](./experiments.zip). The zip-file includes both the original experiments, as well as the extended version after running additional local HPOs for CIFAR-10 dataset with quantity skew and batch size = 8.

## Python files

### [`agg_funcs.py`](./agg_funcs.py)

Description: File contains the aggregation functions used to derive global HPs from optimal clusters.

Functions:

- `average_params`: Use the mean() function to derive global HPs.
- `average_params_weighted_acc`: Use the weighted mean() function w.r.t. accuracy to derive global HPs.
- `mode_params`: Use the median() function to derive global HPs.

### [`clustering.py`](./clustering.py)

Description: Functions used for clustering clients local HPOs, as well as helper functions for ploting the clusters.

Functions:

- `get_gt_data`: Extract ground truth data from experiments.
- `get_client_data`: Extract clients data from experiments.
- `clustering`: Execute the clustering algorithm on local HPO results (preprocessing included).
- `print_clustering_info`: Print additional clustering information.
- `plot_GT`: Plot ground truth data in 2d/3d.
- `plot_client_clusters`: Plot clients clusters in 2d/3d.
- `plot_points_agains_gt_3d`: Plot derived points agains ground truth plot in 3d.
- `plot_points_agains_gt_2d`: Plot derived points agains ground truth plot in 2d.
- `closest_gt_to_point`: Find closest point in ground truth dataset.
- `get_derived_params`: Derive global HPs based on optimal clusters formed with clients data.

### [`export.py`](./export.py)

Description: Runs the algorithm for all experiments and export the results.

Algorithm:
1. DBSCAN on HPs
2. Rank clusters based on scoring function
3. Perform averaging on HP values in optimal clusters

### [`read_data.py`](./read_data.py)

Description: This file was taken from the [`heuristic`](../../heuristic/read_data.py) part of repo for convenience.

Functions:

- `get_local_res`: Read optimal client hyperparameters from local experiments.
- `get_federated_val_acc`: Read best validation accuracy of federated grid search experiment for specific setting.

### [`scoring_funcs.py`](./scoring_funcs.py)

Description: File contains the scoring functions used to rank clusters.

Functions:

- `scoring_function`: Ranks and returns optimal clusters based on the scoring function mean/min/max/mode.
- `extend_df_with_clustering_info`: Add cluster labels (new dataframe column) to every datapoint in dataframe.
- `print_cluster_info`: Print additional information for a given cluster.

### [`visualize.py`](./visualize.py)

Description: Visualizing client HPs vs. ground truth HPs in 2d and 3d.

## Notebooks

### [`feature_skew.ipynb`](./feature_skew.ipynb)

Description: Initial investigation of DBSCAN-based FL-HPO. All feature skew experiments were analyzed with DBSCAN clustering in order to see how the algorithm behaves.

### [`FLoRA.ipynb`](./FLoRA.ipynb)

Description: Implementation of the [FLoRA framework](https://arxiv.org/abs/2112.08524), all 4 unified loss surfaces were implemented. Moreover, this notebook runs FLoRA on our experimental setup and saves the results.

### [`join_new_experiments.ipynb`](./join_new_experiments.ipynb)

Description: Notebook used for merging .csv files (experiments) together - was used when adding additional experiments for CIFAR-10 qty skew.

### [`results_comparison.ipynb`](./results_comparison.ipynb)

Description: Visualize results (global model validation accuracy) achieved by using global HPs derived with DBSCAN-based FL-HPO,  federated grid search and FLoRA framework.

## Results

The [`results/`](./results/) folder contains the results of the DBSCAN-based FL-HPO, as well as FLoRA framework (on both the original and extended experiments). There is one JSON file for every dataset-skew experiment. This folder also contains plots from the [`results_comparison.ipynb`](./results_comparison.ipynb) notebook.

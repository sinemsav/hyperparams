# IID setting scripts

## Requirements

Check Python version with ```python3 -V```. It must be version 3.6.X or 3.7.X. If not, see [here](https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu).

Run the script to install the federated virtual environment on the server: ```./install_fed_env.sh```. It will install the environment and launch ```jupyter notebook``` in background on port 8887.

To connect with the notebook on your own computer, do a ssh bridge like this: ```ssh -N -f -L localhost:{LOCAL_PORT}:localhost:8887 {server_user}@{server_address}```. Then go to ```localhost:{LOCAL_PORT}``` on your browser.

Requirements installed by the script : ```tensorflow==2.4.1, tensorflow_datasets, pandas, talos```



## fedAvg_gridSearch_IID

This notebook finds the best hyperparameters for each client and averages them. Then it tries ```relu,tanh``` and ```sigmoid``` activation functions with the best averaged hyperparameters and outputs which one is the best fit.

In the last cell, specify the number of clients, dataset name in [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/overview) and gridsearch parameters.

Run the notebook.

The notebook will output the averaged best parameters, the best test accuracy with those parameters amongst the clients and the best activation function.

Complete results of the gridsearch of hyperparameters will be in ```experiment_name+"_res/res"+str(numClients))/``` folder.

## fedAvg_intervalSearch_IID

This notebook follows ```fedAvg_gridSearch_IID``` notebook. Once we've found the best averaged hyperparameters and activation function, it will find the best interval for the approximated activation function.

In the last cell, specify the approximated activation function in ```approx_act_fn```, intervals for the approximated activation function in ```intervals```, the number of clients, dataset name and the hyperparameters in ```intervals_params```.

Run the notebook.

The notebook will output the proportion of each interval that were best for each client as well as the best interval amongst the client, and the best test accuracy amongst the clients with this interval.

Complete results of the gridsearch of intervals will be in ```experiment_name+"_res/intervals_res"+str(numClients))/``` folder.

## ADAMApprox_IID

This notebook follow ```fedAvg_intervalSearch_IID``` notebook.

In the last cell, specify the dataset name, approximated activation function in ```act_fn```, the interval for the approximated activation function in ```interval```, the number of times you want the experiment to be run for each precision level in ```runs```, the number of clients and the hyperparameters.

Specify the precisions you want to test in ```precisions``` (0 for 0 digit after the point, 2 for 2 digits after the point, ..., full for full precision). Here we run the experiment ```runs``` times for each precision to get a meaningful average value of loss, accuracy, presicion and recall.

Run the notebook.

```coeff_range``` is the range the coeff ```1 / (sqrt(v_hat) + epsilon)``` takes under full precision.

Final result is the averaged loss, accuracy, presicion and recall for each precision value.
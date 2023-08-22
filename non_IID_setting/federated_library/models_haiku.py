from fedjax import metrics
from fedjax import create_model_from_haiku, Model
import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np

import federated_library.metrics as custom_metrics


class Dropout(hk.Module):
    """Dropout haiku module."""

    def __init__(self, rate: float = 0.5, seed: int = 0):
        """Initializes dropout module.

        Args:
          rate: Probability that each element of x is discarded. Must be in [0, 1).
        """
        super().__init__()
        self._rate = rate
        self._seed = seed

    def __call__(self, x: jnp.ndarray, is_train: bool):
        if is_train or self._rate == 0.0:
            return hk.dropout(rng=jax.random.PRNGKey(self._seed), rate=self._rate, x=x)
        return x


class EMNISTModel(hk.Module):
    """
    Model for EMNIST and SVHN_CROPPED
    """

    def __init__(self, num_classes, act_fn, interval=None):
        super().__init__()
        self._num_classes = num_classes
        self._act_fn = act_fn
        self._interval = interval

    def __call__(self, x: jnp.ndarray):
        x = hk.Conv2D(output_channels=6, kernel_shape=5, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.AvgPool(window_shape=2, strides=1, padding='VALID')(x)
        x = hk.Conv2D(output_channels=16, kernel_shape=5, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.AvgPool(window_shape=2, strides=1, padding='VALID')(x)
        x = hk.Flatten()(x)
        x = hk.Linear(120)(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.Linear(84)(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.Linear(self._num_classes)(x)
        return x


class MNISTModel(hk.Module):
    """
    Model for MNIST
    """

    def __init__(self, num_classes, act_fn, interval=None):
        super().__init__()
        self._num_classes = num_classes
        self._act_fn = act_fn
        self._interval = interval

    def __call__(self, x: jnp.ndarray):
        x = hk.Conv2D(output_channels=32, kernel_shape=5, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.AvgPool(window_shape=3, strides=1, padding='VALID')(x)
        x = hk.Conv2D(output_channels=32, kernel_shape=5, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.AvgPool(window_shape=2, strides=1, padding='VALID')(x)
        x = hk.Flatten()(x)
        x = hk.Linear(200)(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.Linear(self._num_classes)(x)

        return x


class CIFAR10Model(hk.Module):
    """
    Model for CIFAR-10
    """

    def __init__(self, num_classes, act_fn, interval=None, seed=42):
        super().__init__()
        self._num_classes = num_classes
        self._act_fn = act_fn
        self._interval = interval
        self._seed = seed

    def __call__(self, x: jnp.ndarray, is_train: bool):
        x = hk.Conv2D(output_channels=64, kernel_shape=3, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.Conv2D(output_channels=64, kernel_shape=3, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.AvgPool(window_shape=2, strides=1, padding='VALID')(x)
        x = Dropout(0.2, self._seed)(x, is_train)
        x = hk.Conv2D(output_channels=96, kernel_shape=3, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.Conv2D(output_channels=96, kernel_shape=3, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.AvgPool(window_shape=2, strides=1, padding='VALID')(x)
        x = Dropout(0.3, self._seed)(x, is_train)
        x = hk.Conv2D(output_channels=128, kernel_shape=3, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.Conv2D(output_channels=128, kernel_shape=3, padding='VALID')(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = hk.AvgPool(window_shape=2, strides=1, padding='VALID')(x)
        x = Dropout(0.4, self._seed)(x, is_train)
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        if self._interval:
            x = self._act_fn(x, self._interval)
        else:
            x = self._act_fn(x)
        x = Dropout(0.5, self._seed)(x, is_train)
        x = hk.Linear(self._num_classes)(x)

        return x


def get_model(params, ds_info, custom_model=None) -> Model:
    """ Get default haiku model for mnist, emnist, svhn, cifar10,
        or a custom haiku model

    :param params: parameters dict for the model, especially activation function 'act_fn'
    :param ds_info: dataset information
    :param custom_model: an optional custom haiku model (default is None)
    :return: haiku model
    """

    dataset_name = ds_info['dataset_name']
    num_classes = ds_info['num_classes']
    sample_shape = ds_info['sample_shape']
    seed = ds_info['seed']

    (sample_height, sample_width, sample_channels) = sample_shape

    # Defines the expected structure of input batches to the model. This is used to
    # determine the model parameter shapes.
    HAIKU_SAMPLE_BATCH = {
        'x': np.zeros((1, sample_height, sample_width, sample_channels), dtype=np.float32),
        'y': np.zeros(1, dtype=np.float32)
    }

    def TRAIN_LOSS(b, p): return custom_metrics.unreduced_mse_loss(b['y'], p)

    EVAL_METRICS = {
        'loss': custom_metrics.MSELoss(),
        'accuracy': metrics.Accuracy(),
    }

    def forward_pass(batch, is_train=True):
        if custom_model is None:
            if dataset_name == "mnist":
                return MNISTModel(num_classes, act_fn=params["act_fn"],
                                  interval=params.get("interval"))(batch['x'])
            elif dataset_name == "emnist" or dataset_name == "svhn_cropped":
                return EMNISTModel(num_classes, act_fn=params["act_fn"],
                                   interval=params.get("interval"))(batch['x'])
            elif dataset_name == "cifar10":
                return CIFAR10Model(num_classes, act_fn=params["act_fn"],
                                    interval=params.get("interval"), seed=seed)(batch['x'], is_train)
            else:
                raise ValueError(f"No model for {dataset_name}.")
        else:
            return custom_model(batch['x'])

    transformed_forward_pass = hk.transform(forward_pass)

    return create_model_from_haiku(
        transformed_forward_pass=transformed_forward_pass,
        sample_batch=HAIKU_SAMPLE_BATCH,
        train_loss=TRAIN_LOSS,
        eval_metrics=EVAL_METRICS,
        # is_train determines whether to apply dropout or not.
        train_kwargs={'is_train': True},
        eval_kwargs={'is_train': False})

from fedjax import metrics
import jax.numpy as jnp
import jax
from typing import Optional


def unreduced_mse_loss(targets: jnp.ndarray,
                       preds: jnp.ndarray) -> jnp.ndarray:
    """Returns unreduced mean squared error loss."""
    # Without softmax layer
    # err = jnp.argmax(preds, axis=-1) - targets
    # return jnp.sum(jnp.square(err), axis=-1).astype(jnp.float32)

    num_classes = preds.shape[-1]
    # log_softmax gives better performance, but cannot use with HE
    prob_preds = jax.nn.softmax(preds)
    one_hot_targets = jax.nn.one_hot(targets, num_classes)
    err = one_hot_targets - prob_preds
    return jnp.sum(jnp.square(err), axis=-1)


class MSELoss(metrics.Metric):
    """Metric for mean squared error loss.

    Attributes:
      target_key: Key name in ``example`` for target.
      pred_key: Key name in ``prediction`` for unnormalized model output pred.
    """
    target_key: str = 'y'
    pred_key: Optional[str] = None

    def zero(self) -> metrics.MeanStat:
        return metrics.MeanStat.new(0., 0.)

    def evaluate_example(self, example: metrics.SingleExample,
                         prediction: metrics.SinglePrediction) -> metrics.MeanStat:
        """Computes cross entropy loss for a single example.

        Args:
          example: One example with target in range [0, num_classes) of shape [1].
          prediction: Unnormalized prediction for ``example`` of shape [num_classes]

        Returns:
          MeanStat for loss for a single example.
        """
        target = example[self.target_key]
        pred = prediction if self.pred_key is None else prediction[self.pred_key]
        loss = unreduced_mse_loss(target, pred)
        return metrics.MeanStat.new(loss, 1.)

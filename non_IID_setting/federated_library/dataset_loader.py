import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.utils import to_categorical
from federated_library.display_distribution import display_dataset_barplot


def load_tf_dataset(dataset_name: str, decentralized, skew_type="", display=True):
    """ Load `dataset_name` dataset from tensorflow

    :param dataset_name: dataset name from tensorflow
    :param decentralized: True if running decentralized experiment
    :param skew_type: skew type, can be ['qty', 'label', 'feature', None]
    :param display: True to display the dataset's labels distribution(default is True)
    :return: (x_train, y_train) training split, (x_test, y_test) test split,
                ds_info dataset information containing
                [dataset_name, num_classes, num_clients, sample_shape, skew_tye, seed]
    """

    ds, ds_info = tfds.load(dataset_name, split=['train', 'test'], batch_size=-1, as_supervised=True,
                            with_info=True)

    (x_train, y_train), (x_test, y_test) = tfds.as_numpy(ds)

    ds_info = dict(dataset_name=dataset_name,
                   num_classes=ds_info.features['label'].num_classes,
                   num_clients=None,
                   sample_shape=ds_info.features['image'].shape,
                   skew_type=skew_type,
                   seed=7)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # shuffle both train and test once (to average between runs later on..)
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    y_test = y_test[shuffler]

    if display:
        display_dataset_barplot(y_train, ds_info['num_classes'], "Train split")
        display_dataset_barplot(y_test, ds_info['num_classes'], "Test split")

    if decentralized:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test), ds_info

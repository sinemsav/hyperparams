from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, AveragePooling2D


def get_model(params, ds_info, custom_model=None):
    """ Get default keras model for mnist, emnist, svhn or cifar10
        or a custom keras model

    :param params: parameters dict for the model, especially activation function 'act_fn'
    :param ds_info: dataset information
    :param custom_model: an optional custom keras model (default is None)
    :return: keras model
    """
    dataset_name = ds_info['dataset_name']
    num_classes = ds_info['num_classes']
    sample_shape = ds_info['sample_shape']

    if custom_model is not None:
        return custom_model

    # emnist and svhn
    if dataset_name == "emnist" or dataset_name == "svhn_cropped":
        model = Sequential()
        model.add(Conv2D(16, (5, 5), activation=params['act_fn'], input_shape=sample_shape, name="1"))
        model.add(AveragePooling2D((2, 2), name="2"))
        model.add(Conv2D(6, (5, 5), activation=params['act_fn'], name="3"))
        model.add(AveragePooling2D((2, 2), name="4"))
        model.add(Flatten())
        model.add(Dense(120, activation=params['act_fn'], name="5"))
        model.add(Dense(84, activation=params['act_fn'], name="6"))
        model.add(Dense(num_classes, activation='softmax'))
        return model
    # mnist
    elif dataset_name == "mnist":
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation=params['act_fn'], input_shape=sample_shape, name="1"))
        model.add(AveragePooling2D((3, 3), name="2"))
        model.add(Conv2D(32, (5, 5), activation=params['act_fn'], name="3"))
        model.add(AveragePooling2D((2, 2), name="4"))
        model.add(Flatten(name="5"))
        model.add(Dense(200, activation=params['act_fn'], name="6"))
        model.add(Dense(num_classes, activation='softmax', name="7"))
        return model
    # cifar10
    elif dataset_name == "cifar10":
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation=params['act_fn'], kernel_initializer='he_uniform', padding='same',
                         input_shape=sample_shape, name='1'))
        model.add(
            Conv2D(64, (3, 3), activation=params['act_fn'], kernel_initializer='he_uniform', padding='same', name='2'))
        model.add(AveragePooling2D((2, 2), name='3'))
        model.add(Dropout(0.2, name='4'))
        model.add(
            Conv2D(96, (3, 3), activation=params['act_fn'], kernel_initializer='he_uniform', padding='same', name='5'))
        model.add(
            Conv2D(96, (3, 3), activation=params['act_fn'], kernel_initializer='he_uniform', padding='same', name='6'))
        model.add(AveragePooling2D((2, 2), name='7'))
        model.add(Dropout(0.3, name='8'))
        model.add(
            Conv2D(128, (3, 3), activation=params['act_fn'], kernel_initializer='he_uniform', padding='same', name='9'))
        model.add(Conv2D(128, (3, 3), activation=params['act_fn'], kernel_initializer='he_uniform', padding='same',
                         name='10'))
        model.add(AveragePooling2D((2, 2), name='11'))
        model.add(Dropout(0.4, name='12'))
        model.add(Flatten(name='13'))
        model.add(Dense(128, activation=params['act_fn'], kernel_initializer='he_uniform', name='14'))
        model.add(Dropout(0.5, name='16'))
        model.add(Dense(num_classes, activation='softmax', name='15'))
        return model

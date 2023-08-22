def average_params(ind_list, df):
    """Get averaged parameter (learning rate, momentum) values of data at ind_list position in dataframe df.
    For batch size mode is used.

    Args:
        ind_list (list): list of indices denoting the position in dataframe
        df (pandas.DataFrame): all datapoints in dataframe

    Returns:
        (float, float, float): average value for learning rate, momentum and mode for batchsize
    """
    return (
        df.iloc[ind_list].client_lr.mean(),
        df.iloc[ind_list].client_mom.mean(),
        df.iloc[ind_list].batch_size.mode()[0],
    )


def average_params_weighted_acc(ind_list, df):
    """Get weighted averaged parameter (learning rate, momentum) values of data at ind_list position in dataframe df.
    The parameters are weighted by achieved validation accuracy. For batch size mode is used.

    Args:
        ind_list (list): list of indices denoting the position in dataframe
        df (pandas.DataFrame): all datapoints in dataframe

    Returns:
        (float, float, float): weighted average value for learning rate, momentum and mode for batchsize
    """
    dfN = df.iloc[ind_list]
    total_acc = dfN.accuracy.sum()

    dfN["w_lr"] = dfN.apply(
        lambda row: row.client_lr * row.accuracy / total_acc, axis=1
    )
    dfN["w_mom"] = dfN.apply(
        lambda row: row.client_mom * row.accuracy / total_acc, axis=1
    )

    return dfN["w_lr"].sum(), dfN["w_mom"].sum(), dfN.batch_size.mode()[0]


def mode_params(ind_list, df):
    """Get mode parameter (learning rate, momentum, batch size) values of data at ind_list position in dataframe df.

    Args:
        ind_list (list): list of indices denoting the position in dataframe
        df (pandas.DataFrame): all datapoints in dataframe

    Returns:
        (float, float, float): mode value for learning rate, momentum and batchsize
    """
    dfN = df.iloc[ind_list]

    return dfN.client_lr.mode()[0], dfN.client_mom.mode()[0], dfN.batch_size.mode()[0]

import numpy as np
from keras.preprocessing import timeseries_dataset_from_array
import tensorflow as tf

# this is old for non-darts timeseries and dataprocessing
def create_dataset_from_df(df, source_window_size: int, rolling_window_in_ptus: int = 96) -> tf.data.Dataset:
    return timeseries_dataset_from_array(
        data=df,
        targets=None,
        sequence_length=source_window_size,  # training/predict window size
        sequence_stride=rolling_window_in_ptus,  # rolling window
        sampling_rate=1,  # every nth item
        batch_size=1,
        shuffle=False,
        seed=42
    )


def create_dataset_from_clustered_df(clustered_df, dataset_size, rolling_window_size):
    """
    Creates a BatchDataset that can be used for training, tailored towards timeseries data, does not include X,y-splitting
    :param clustered_df: df.groupby(clustering_key)
    :param dataset_size: size of one training sample (input_size+target_size) (since we will be doing that split later
    :param rolling_window_size:
    :return: a tf.data.BatchDataset, can be converted to numpy array with .as_numpy_iterator for further processing
    """
    dataset = None
    for _, group in clustered_df:
        if dataset is None:
            dataset = create_dataset_from_df(group, dataset_size, rolling_window_size)
        else:
            dataset = dataset.concatenate(create_dataset_from_df(group, dataset_size, rolling_window_size))
    return dataset


def create_X_y_split(data: np.ndarray, target_size: int, target_column_index: int):
    """
    creates the split for training a model of X, y, where X can be a multidimensional array/vector, and y is a one dimensional array
    :param data: numpy array used for training in expected shapes
                of (no of samples, timeseries length, number of features) or (timeseries length, number of features)
    :param target_size: size of how many datapoints we want to predict and train for predicting
    :param target_column_index: this numpy array originated somewhere from a dataframe at somepoint,
                                so all non-predictor variables we need to filter out and
                                we need to know which column (index of column name) we want to keep
    :return: tuple of X, y as numpy-arrays
    """
    if len(data.shape) == 2:
        return data[:-target_size], data[-target_size:, target_column_index]
    elif len(data.shape) == 3:
        return data[:, :-target_size], data[:, -target_size:, target_column_index]
    else:
        raise Exception(f"Unexpected shape got: {data.shape} but expect something along (3, a, b) or (2, a, b)")

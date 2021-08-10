import pandas as pd
import random
from typing import List, Tuple



def shuffle_data(data: List[Tuple[List[float], float]]) -> List[Tuple[List[float], float]]:
    '''
    Used for shuffling the data during the training/validation phases.
    NOTE: Param data is a Python list.
    '''
    size = len(data)
    for row_ind in range(size):
        swap_row_ind = random.randrange(size)
        tmp_row = data[swap_row_ind]
        data[swap_row_ind] = data[row_ind]
        data[row_ind] = tmp_row

    return data



def generate_dataset(data: pd.DataFrame, limit: int, offset: int, data_aug_per_sample: int = 0) -> List[Tuple[List[float], float]]:
    '''
    Returns a list of tuples, of which the first element of the tuple is the list of values for the features and the second is the target value
    NOTES:
    - data_aug_per_sample param determines how many extra datapoints to generate per each original datapoint * its frequency metric (i.e., signal_ratios)
    - signal_ratios variable is used to upsample underrepresented categories more than their counterparts when augmenting the data
    '''
    # to determine relative frequency of signals
    new_data = data.iloc[:limit,:]
    vals = new_data["signal"].value_counts().sort_index()
    signal_ratios = [vals.max()/x for x in vals]

    dataset = []
    for row in range(offset, limit):
        target = data.iloc[row, -1]
        row_features = []

        # -1 excludes the signal column
        for feature in range(data.shape[1] - 1):
            row_features.append(data.iloc[row, feature])
        datapoint_tuple = (row_features, target)
        dataset.append(datapoint_tuple)

        # this evens out the datapoints per category
        for i in range(data_aug_per_sample * round(signal_ratios[target])):
            row_features_aug = []
            # -1 excludes the signal column
            for feature in range(data.shape[1] - 1):
                rand_factor = 1 + random.uniform(-0.000001, 0.000001)
                row_features_aug.append(data.iloc[row, feature] * rand_factor)
            datapoint_tuple_aug = (row_features_aug, target)
            dataset.append(datapoint_tuple_aug)

    return dataset



def load_data(coin: str) -> pd.DataFrame:
    '''
    Loads relevant data for given coin.
    '''
    data = pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv")
    data = data.drop(columns=["date"])
    data["signal"] = data["signal"].astype("int64")

    return data



def check_if_data_is_clean(data: pd.DataFrame) -> None:
    '''
    Checks for any anomalous, unnormalized data in all columns except the signal column.
    '''
    for c in range(len(data.columns)-1):
        for r in range(len(data)):
            if (data.iloc[r, c] > 1):
                raise Exception(f"Data unfit for processing! Unnormalized data still present in the dataset in column = {data.columns[c]}, row = {r}.")



def get_datasets(coin: str, data_aug_factor: int = 0) -> Tuple[List[Tuple[List[float], float]], List[Tuple[List[float], float]], List[Tuple[List[float], float]]]:
    '''
    Splits dataset into training, validation, and testing datasets.
    NOTE: uses no data augmentation by default and will only apply data_aug_factor to the training dataset.
    '''
    data = load_data(coin)
    try:
        check_if_data_is_clean(data)
    except:
        raise

    # Split into training, validation, testing
    # 70-15-15 split
    n_datapoints = data.shape[0]
    train_end = int(round(n_datapoints*0.7))
    valid_end = train_end + int(round(n_datapoints*0.15))

    train_data = generate_dataset(data, train_end, 0, data_aug_factor)
    print("Training dataset created.")

    valid_data = generate_dataset(data, valid_end, train_end)
    print("Validation dataset created.")

    test_data = generate_dataset(data, n_datapoints, valid_end)
    print("Testing dataset created.")

    return train_data, valid_data, test_data




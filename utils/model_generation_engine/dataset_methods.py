import os
import pandas as pd
import random
import time
import torch
from . import neural_nets as nn
from typing import List, Tuple

#
# ---------- HELPER METHODS ----------
#
def convert_to_tensor(model: nn.CryptoSoothsayer, features: List[float], target: float) -> Tuple[torch.tensor, torch.tensor]:
    '''
    Converts the feature vector and target into pytorch-compatible tensors.
    '''
    feature_tensor = torch.tensor([features], dtype=torch.float32)
    feature_tensor = feature_tensor.to(model.get_device())
    target_tensor = torch.tensor([target], dtype=torch.int64)
    target_tensor = target_tensor.to(model.get_device())

    return feature_tensor, target_tensor



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



def check_if_data_is_clean(data: pd.DataFrame) -> None:
    '''
    Checks for any anomalous, unnormalized data in all columns except the signal column.
    '''
    for c in range(len(data.columns)-1):
        for r in range(len(data)):
            if (data.iloc[r, c] > 1):
                raise Exception(f"Data unfit for processing! Unnormalized data still present in the dataset in column = {data.columns[c]}, row = {r}.")



def load_data(coin: str) -> pd.DataFrame:
    '''
    Loads relevant data for given coin.
    '''
    data = pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv")
    data = data.drop(columns=["date"])
    data["signal"] = data["signal"].astype("int64")

    return data



#
# ---------- DATASET CREATION ----------
#
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



def prepare_model_pruning_datasets(coin: str) -> Tuple[List[Tuple[List[float], float]], List[Tuple[List[float], float]], List[Tuple[List[float], float]]]:
    start_time = time.time()
    data_aug_factor = 0
    print("Creating datasets...")

    # separate valid, test
    try:
        train_data, valid_data, test_data = get_datasets(coin, data_aug_factor)
    except:
        raise

    # all data together
    data = train_data + valid_data + test_data

    print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")

    return data, valid_data, test_data



#
# ---------- DATASET MERGING ----------
#
def merge_datasets(coin: str, list_of_datasets: List[pd.DataFrame], all_data: bool = False) -> None:
    '''
    Merges two or more datasets.
    Param all_data is used when combining all datasets into one mega dataset.

    NOTE: Param list_of_datasets must be a list of pandas DataFrames
    '''
    merged_data = pd.concat(list_of_datasets)
    merged_data["date"] = pd.to_datetime(merged_data["date"], dayfirst=True, infer_datetime_format=True)
    merged_data = merged_data.sort_values(by=["date"], ascending=False)
    # if merging all datasets into mega training dataset, more than date must be unique
    if all_data:
        subset_cols = ["date", "price", "market_cap", "volume"]
    else:
        subset_cols = ["date"]

    merged_data = merged_data.drop_duplicates(subset=subset_cols, keep="last")
    merged_data = merged_data.reset_index()
    merged_data = merged_data.drop(columns=["index"])

    if all_data:
        merged_data.to_csv(f"datasets/complete/all_historical_data_complete.csv", index=False, float_format="%f")
    else:
        merged_data.to_csv(f"datasets/raw/{coin}_historical_data_raw.csv", index=False, float_format="%f")



def merge_new_dataset_with_old(coin: str, by_range: bool = True) -> str:
    '''
    Merges all previous datasets with the newly fetched data.
    NOTE: Assumes fetch_missing_data_by_range or fetch_missing_data_by_date have been called first.
    '''
    data_to_merge = [pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")]

    if by_range:
        data_to_merge.append(pd.read_csv(f"datasets/raw/{coin}_historical_data_by_range.csv"))
        os.remove(f"datasets/raw/{coin}_historical_data_by_range.csv")
    else:
        data_to_merge.append(pd.read_csv(f"datasets/raw/{coin}_historical_data_by_date.csv"))
        os.remove(f"datasets/raw/{coin}_historical_data_by_date.csv")

    merge_datasets(coin, data_to_merge)

    return f"Successfully merged new and old {coin} data"



def combine_all_datasets(coins: List[str]) -> None:
    '''
    Combines all datasets that have already been prepared for training.
    '''
    datasets_to_merge = []
    for coin in coins:
        # Currently polkadot has fewer than 350 days worth of datasets
        # In the future, I can delete this line
        if coin != "polkadot":
            datasets_to_merge.append(pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv"))

    merge_datasets("all", datasets_to_merge, all_data=True)



#
# ---------- DATASET CLEANER ----------
#
def remove_greater_than_one_artifacts(data_type: str) -> None:
    '''
    The purpose of this method is to clean up some strange artifacts that found their way into the datasets where numbers that are supposed to be less than 1 are somehow larger by several orders of magnitude.

    The source of this error remains unknown.
    '''
    for coin in common.coins:
        data = pd.read_csv(f"datasets/{data_type}/{coin}_historical_data_{data_type}.csv")

        count = 0
        for c in range(1, len(data.columns)):
            for r in range(1, len(data)-1):
                former_cell = data.iloc[r-1, c]
                curr_cell = data.iloc[r, c]
                latter_cell = data.iloc[r+1, c]
                if (curr_cell / 10) > former_cell or (curr_cell / 10) > latter_cell:
                    factor = curr_cell // min(former_cell, latter_cell)
                    print(c,r, factor)
                    if factor >= 100:
                        data.iloc[r, c] = data.iloc[r, c] / 1000
                    elif factor >= 10:
                        data.iloc[r, c] = data.iloc[r, c] / 100
                    count += 1

        print(count)

        data.to_csv(f"datasets/{data_type}/{coin}_historical_data_{data_type}.csv", index=False, float_format="%f")



def clean_for_training() -> None:
    '''
    Takes a clean dataset and prunes it so that it's suitable for training.
    '''
    for coin in common.coins:
        data = pd.read_csv(f"datasets/clean/{coin}_historical_data_clean.csv")

        # find first instance of real SMA_350 value
        data = data.iloc[1: , :] # drop first row as it will always be normalized to 1
        start_ind = data[data["price_350_SMA"] == 1].first_valid_index()
        if start_ind != None:
            data = data.iloc[start_ind: , :]

            # find last instance of signal value and trim dataset
            data = data.iloc[:len(data)-common.SIGNAL_FOR_N_DAYS_FROM_NOW, :]
            data.to_csv(f"datasets/complete/{coin}_historical_data_complete.csv", index=False, float_format="%f")
        else:
            print(f"{coin} does not have enough data.")

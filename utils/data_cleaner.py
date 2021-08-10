import pandas as pd
from . import common


def remove_greater_than_one_artifacts(data_type: str) -> None:
    '''
    The purpose of this file is to clean up some strange artifacts that found their way into the datasets where numbers that are supposed to be less than 1 are somehow larger by several orders of magnitude.

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

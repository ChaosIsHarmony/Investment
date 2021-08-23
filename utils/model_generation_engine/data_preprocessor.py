import pandas as pd
from .. import common
from typing import Dict, List, Tuple



def handle_missing_data(coin: str, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    '''
    Checks for missing days
    Fills all NaN values with 0.
    Takes average of prior day and next day to calculate missing value or previous or next day if data point is at the beginning or end of the dataset, respectively.
    '''
    # check for missing dates
    data["date"] = pd.to_datetime(data["date"])
    missing_dates = pd.date_range(start = start_date, end = end_date ).difference(data["date"])
    if len(missing_dates) > 0:
        print("There were missing dates in the dataset. Fetching missing data.")
        common.fetch_missing_data_by_dates(coin, missing_dates, verbose=True)
        common.merge_newly_aggregated_data(coin, by_range=False)
        print("Missing dates successfuly fetched.")
        return None

    data = data.fillna(0)

    for i, row in data.iterrows():
        for column in data.columns[1:]:
            if row[column] == 0:
                next_non_zero = 0
                start_ind = i + 1
                while next_non_zero == 0 and start_ind < data.shape[0]:
                    next_non_zero += data.loc[start_ind, column]
                    start_ind += 1

                # leading zero
                if i == 0:
                    data.loc[i, column] = next_non_zero
                # Take average of two closest data points
                elif next_non_zero > 0:
                    data.loc[i, column] = (data.loc[i-1, column] + next_non_zero) / 2
                # Otherwise, just make same as one before it
                elif i > 0:
                    data.loc[i, column] = data.loc[i-1, column]

    return data



def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalizes data using min-max normalization but only up until the given point in history, e.g., datapoint for 2021/02/28 does not have any knowledge of data from 01/03/2021 and onwards.
    NOTE: only uses 2nd until 2nd to last column, because 1st column is the date and the last column is the signal (i.e., what the appropriate action to take should be).
    '''
    data_cp = data.copy(deep=True)

    # 1st row items only have themselves as history
    for column in range(1, data.shape[1]-1):
        data_cp.iloc[0, column] = 1.0

    for i in range(1, data.shape[0]):
        for column in range(1, data.shape[1]-1):
            col_max = data.iloc[:i+1, column].max()
            col_min = data.iloc[:i+1, column].min()
            # to avoid division by zero
            if col_max == col_min and col_max != 0:
                col_min -= 1
            elif col_max == 0:
                col_max += 1
            data_cp.iloc[i, column] = (data.iloc[i, column] - col_min) / (col_max - col_min)

    # last row
    for column in range(1, data.shape[1]-1):
        col_max = data.iloc[:, column].max()
        col_min = data.iloc[:, column].min()
        if col_max == col_min and col_max != 0:
            col_min -= 1
        elif col_max == 0:
            col_max += 1
        data_cp.iloc[-1, column] = (data.iloc[-1, column] - col_min) / (col_max - col_min)

    # fear and greed index is out of 100
    for column in data.columns:
        if "fear_greed" in str(column):
            data_cp[column] = data[column] / 100

    # in case there were division by zero errors leading to NaN
    data_cp = data_cp.fillna(0)

    return data_cp



def perform_SMA_calculation(data: pd.DataFrame, totals: Dict[int, float], SMAs: Dict[int, List[float]], column_name: str) -> pd.DataFrame:
    for i, row in data.iterrows():
        for key in totals.keys():
            totals[key] += data.loc[i, column_name]
            # can't do i+1 because (i+1)-key would leave only 4 days in the total for the 5-day MA, 9 for the 10-day, etc. and would make the calculation inaccurate
            if key <= i:
                totals[key] -= data.loc[i-key, column_name]
                SMAs[key][i] = totals[key] / key

    for key in SMAs.keys():
        data[f"{column_name}_{key}_SMA"] = SMAs[key]

    return data



def generate_price_SMA_lists_dict(list_size: int) -> Dict[int, List[float]]:
    SMAs = {5: [], 10:[], 25: [], 50: [], 75: [], 100: [], 150: [], 200: [], 250: [], 300: [], 350: []}

    for i in range(list_size):
        for key in SMAs.keys():
            SMAs[key].append(0.0)

    return SMAs



def generate_fear_greed_SMA_lists_dict(list_size: int) -> Dict[int, List[float]]:
    SMAs = {3: [], 5:[], 7: [], 9: [], 11: [], 13: [], 15: [], 30: []}

    for i in range(list_size):
        for key in SMAs.keys():
            SMAs[key].append(0.0)

    return SMAs



def calculate_price_SMAs(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates Simple Moving Averages to the maximum extent allowed by the data
    '''
    n_datapoints = data.shape[0]
    totals = {5:0, 10:0, 25:0, 50:0, 75:0, 100:0, 150:0, 200:0, 250:0, 300:0, 350:0}
    SMAs = generate_price_SMA_lists_dict(n_datapoints)

    data = perform_SMA_calculation(data, totals, SMAs, "price")

    return data



def calculate_fear_greed_SMAs(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates Simple Moving Averages for Fear/Greed index over several discrete intervals for the past fortnight
    '''
    n_datapoints = data.shape[0]
    totals = {3:0, 5:0, 7:0, 9:0, 11:0, 13:0, 15:0, 30:0}
    SMAs = generate_fear_greed_SMA_lists_dict(n_datapoints)

    data = perform_SMA_calculation(data, totals, SMAs, "fear_greed")

    return data



def calculate_SMAs(data: pd.DataFrame) -> pd.DataFrame:
    # reverse the dataframe for easier calculation logic
    data = data.reindex(index=data.index[::-1]).reset_index()
    data = data.drop(columns=["index"])

    # price
    data = calculate_price_SMAs(data)
    # fear/greed index
    data = calculate_fear_greed_SMAs(data)

    return data



def get_signal_value(percent_delta: float) -> int:
    '''
    Returns signal to BUY, SELL, or HODL [0-3 scale] based on percent_delta over given period (as determined by the calling method).
    '''
    signal = 1 # HODL by default

    # SELL - lower is stronger
    if percent_delta < -0.175:
        signal = 2
    # BUY - higher is stronger
    elif percent_delta > 0.15:
        signal = 0

    return signal



def get_weighting_constant(n: int = 28) -> float:
    '''
    Calculates weighting constant required for the given period such that:

        The sum from j=1 to j=n of (weight_constant * j) = 1

            e.g., when n = 2, then
            (wc * 1) + (wc * 2) = 1 and, thus wc = 1/3

    NOTE: simplified from (100 / ((n * (n+1)) / 2)) / 100, where the numerator provides a constant that when multiplied by unit increments, the addition of all terms from 1 -> n will add to 1.
    '''
    return 2 / (n*(n+1))



def calculate_signals(data: pd.DataFrame, interval: int = 28, verbose: bool = False) -> pd.DataFrame:
    '''
    Calculates the signal on a scale from 0-3 (BUY, HODL, & SELL) based on weighted average of the price future (days_out) price movement deltas.
    If percentage increase exceeds given threshold, then SELL; if decrease then BUY.
    If percentage increase/decrease does not exceed minimum thresholds, then HODL.
    NOTE: The calculation weights days in the more distant future more heavily, as they are closer to what the actual value will be at the end of the specified time interval.
    '''
    signals = []
    weighting_constant = get_weighting_constant(interval)
    for ind in range(len(data) - interval):
        current_price = data["price"][ind]
        price_delta_avg = 0.0

        for days_from_now in range(1, interval+1):
            later_price = data["price"][ind+days_from_now]
            percent_delta = (later_price - current_price) / current_price
            price_delta_avg += percent_delta * weighting_constant * days_from_now

        signals.append(get_signal_value(price_delta_avg))

    data["signal"] = pd.Series(signals)

    if verbose:
        print("Value counts for signals in dataset:\n", data["signal"].value_counts())

    data = data.fillna(0)

    return data



def clean_data(coin: str, data: pd.DataFrame, start_date: str, end_date: str, verbose=False) -> pd.DataFrame:
    '''
    Preprocesses the basic data provided by coingecko in the following ways:

        - Fills in missing values
        - Calculates Simple Moving Averages for a variety of intervals
        - Calculates the signal for that day
            - Prescient looking forward x-days and averaging the price_deltas
        - Normalizes all values by dividing by the max value in each category
            - Normalizes neither date nor signal columns
    '''
    # Fill in missing values
    data = handle_missing_data(coin, data, start_date, end_date)
    if data is None:
        return None
    if verbose:
        print(f"Missing data handling complete for {coin}.")
    # Calculate SMAs
    data = calculate_SMAs(data)
    if verbose:
        print(f"SMA calculation complete for {coin}.")
    # Calculate signals
    data = calculate_signals(data, common.SIGNAL_FOR_N_DAYS_FROM_NOW)
    if verbose:
        print(f"Signal calculation for {common.SIGNAL_FOR_N_DAYS_FROM_NOW} days from now complete for {coin}.")
    # save all features raw file for use in signal_generator
    data.to_csv(f"datasets/raw/{coin}_historical_data_raw_all_features.csv", index=False, float_format="%f")
    # Normalize, must happen after SMA calculation or will skew results
    data = normalize_data(data)
    if verbose:
        print(f"Data normalization complete for {coin}.")
    print()

    return data



if __name__ == "__main__":
    #  coins = ["algorand", "bitcoin", "cardano", "chainlink", "ethereum", "polkadot", "solana"]
    # The following two coins have shorter histories and require a different start date {polkadot = 2020-08-23; solana = 2020-04-11}
    coins = ["avalanche-2"]
    #coins = ["polkadot"]
    #coins = ["solana"]
    start_date = "2020-09-22"#"2020-08-23"#"2019-10-20"
    yesterday = date.today() - timedelta(1)
    end_date = yesterday

    for coin in coins:
        complete = False
        while not complete:
            print(coin)
            data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")
            data = clean_data(coin, data, start_date, end_date, verbose=True)
            if data is None:
                print(f"The dataset was missing dates. Used utils/data_aggregator.py to collect the missing dates. Beginning preprocessing again.")
            else:
                data.to_csv(f"datasets/clean/{coin}_historical_data_clean.csv", index=False, float_format="%f")
                complete = True

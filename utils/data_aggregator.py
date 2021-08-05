'''
USED AS PART OF THE PROCESS TO CREATE A DATASET WITH WHICH TO TRAIN SUPERVISED LEARNING ML ALGORITHMS.

FUNCTION: TO PULL DATA FROM COINGECKO DATABASE AND STORE AS A CSV.

NOTE: THE 'SIGNAL' COLUMN IS FILLED IN BY A HUMAN IN HINDSIGHT WITH THE CORRECT ACTION GIVEN THE STATE OF THE MARKET AT THE TIME.
'''
import os
import requests
import json
from datetime import date, datetime, timedelta
import time
import pandas as pd
import numpy as np
import common
from typing import List


def get_fear_greed() -> dict:
    '''
    Pulls the data for the fear and greed index at the time it's called.
    Returns an int.
    '''
    return int(requests.get("https://api.alternative.me/fng/?date_format=cn").json()["data"][0]["value"])



def get_fear_greed_by_range(n_days: int) -> dict:
    return requests.get(f"https://api.alternative.me/fng/?limit={n_days}&date_format=cn").json()["data"]



def get_historic_data(coin: str, date: str) -> dict:
    '''
    Pulls all data from coingecko for specified coin on specified date.
    Returns a dictionary.
    '''
    return requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/history?date={date}").json()



def get_correct_date_format(date: str) -> str:
    '''
    Puts the Python datetime into a formatted string the coingecko api finds more copacetic i.e., dd-mm-yyyy.
    NOTE: Param date must be a datetime object.
    '''
    well_formed_date = ""
    if date.day < 10:
        well_formed_date += "0" + str(date.day) + "-"
    else:
        well_formed_date += str(date.day) + "-"

    if date.month < 10:
        well_formed_date += "0" + str(date.month) + "-"
    else:
        well_formed_date += str(date.month) + "-"

    well_formed_date += str(date.year)

    return well_formed_date



def get_generic_score(data: dict) -> float:
    '''
    Extracts score for given field from data.
    '''
    score = 0

    for key in data.keys():
        if data[key]:
            try:
                score += float(data[key])
            except Exception:
                print("Bad value when extracting score.")

    return score



def extract_basic_data(data: dict, date: str) -> dict:
    '''
    Extracts all useful information from the coingecko data.
    '''
    data_dict = {}

    data_dict["date"] = date

    if "market_data" in data.keys():
        data_dict["price"] = data["market_data"]["current_price"]["twd"]
        data_dict["market_cap"] = data["market_data"]["market_cap"]["twd"]
        data_dict["volume"] = data["market_data"]["total_volume"]["twd"]
    else:
        data_dict["price"] = 0
        data_dict["market_cap"] = 0
        data_dict["volume"] = 0

    return data_dict



def get_time() -> int:
    '''
    Returns current time rounded to milliseconds.
    NOTE: time.time() returns current time as a floating point.
    '''
    return int(round(time.time() * 1000))



def fetch_missing_data_by_dates(coin: str, dates: List[str], verbose: bool = False) -> pd.DataFrame:
    '''
    WARNING: Cannot automatically fetch Fear/Greed index <- This is partially alleviated by the data_preprocessors handle_missing_data method.
    '''
    historical_data = []
    missing_dates = []

    for date in dates:
        try:
            data = get_historic_data(coin, date)
        except:
            print(f"Error on {date}")
            missing_dates.append(date)
            continue

        historical_data.append(extract_basic_data(data, date))

        # To help regulate the speed with which calls are being made
        # to assuage 434 return codes
        time.sleep(1)

    if len(missing_dates) > 0:
        ans = input("Try again? [y/n] ")
        if (ans.lower())[0] == 'y':
            more_data = fetch_missing_data_by_dates(coin, missing_dates)
            # merge the data
            historical_data = pd.DataFrame(historical_data)
            historical_data = pd.concat([historical_data, more_data])

    # save as CSV
    coin_data = pd.DataFrame(historical_data)
    coin_data.to_csv(f"datasets/raw/{coin}_historical_data_by_date.csv", index=False, float_format="%f")

    if verbose:
        print(f"{coin} data successfully pulled and stored.")

    return coin_data



def fetch_missing_data_by_range(coin: str, n_days: int, start_delta: int = 0, verbose: bool = False) -> str:
    today = date.today() - timedelta(start_delta)
    historical_data = []
    missing_dates = []
    fear_greed = get_fear_greed_by_range(n_days)
    fear_greed_ind = 0

    for i in range(n_days):
        next_date = get_correct_date_format(today - timedelta(i))
        try:
            data = get_historic_data(coin, next_date)
        except:
            print(f"Error on {next_date}")
            missing_dates.append(next_date)
            continue

        daily_data = extract_basic_data(data, next_date)
        daily_data["fear_greed"] = fear_greed[fear_greed_ind]["value"]
        fear_greed_ind += 1
        historical_data.append(daily_data)

        # To help regulate the speed with which calls are being made
        # to assuage 434 return codes
        time.sleep(1)

    # if there's still missing data
    if len(missing_dates) > 0:
        ans = input("Try again? [y/n] ")
        if (ans.lower())[0] == 'y':
            more_data = fetch_missing_data_by_dates(coin, missing_dates)
            # merge the data
            historical_data = pd.DataFrame(historical_data)
            historical_data = pd.concat([historical_data, more_data])

    # save as CSV
    coin_data = pd.DataFrame(historical_data)
    coin_data.to_csv(f"datasets/raw/{coin}_historical_data_by_range.csv", index=False, float_format="%f")

    message = f"{coin} data successfully pulled and stored."
    if verbose:
        print(message)

    return message



def aggregate_data_for_new_coins(coins: List[str], how_far_back: int = 600) -> None:
    '''
    Param coins is a list of all the coins to aggregate data for.
    Param how_far_back indicates how many days counting backwards from today to collect data for.
    '''
    today = date.today()
    api_calls = 0
    api_call_cycle_start = get_time()
    fear_greed = get_fear_greed_by_range(how_far_back)

    for coin in coins:
        date_delta = -1
        fear_greed_ind = 0
        has_next = True
        missing_dates = []

        # Extract basic data
        historical_data = []
        while has_next:
            # There is a limit of 100 api calls per minute
            # But regularly returns a 434 even with much lower calls/minute
            api_calls += 1
            if api_calls > 70:
                time_to_wait = 60 - ((get_time() - api_call_cycle_start) / 1000)
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    print(f"Slept for {time_to_wait} seconds.")

                    api_call_cycle_start = get_time()
                    api_calls = 1

                # Request data
                date_delta += 1
                if date_delta >= how_far_back:
                    has_next = False
                    continue

                next_date = get_correct_date_format(today - timedelta(date_delta))

                try:
                    data = get_historic_data(coin, next_date)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Coin: {coin}")
                    print(f"Date that failed: {next_date}")
                    print(f"Days from today: {date_delta}")
                    missing_dates.append(next_date)
                    continue

                daily_data = extract_basic_data(data, next_date)
                daily_data["fear_greed"] = fear_greed[fear_greed_ind]["value"]
                fear_greed_ind += 1

                historical_data.append(daily_data)

            # save as CSV
            coin_data = pd.DataFrame(historical_data)
            coin_data.to_csv(f"datasets/raw/{coin}_historical_data_raw.csv", index=False, float_format="%f")

            # if missing dates
            if len(missing_dates) > 0:
                fetch_missing_data_by_dates(coin, missing_dates, verbose=True)
                common.merge_newly_aggregated_data(coin, by_range=False)

            print(f"{coin} data successfully pulled and stored.")



if __name__ == "__main__":
    #aggregate_data_for_new_coins(common.coins)
    aggregate_data_for_new_coins(common.possible_coins)

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

coin_id = ["aave", "algorand", "bitcoin", "cardano", "chainlink", "cosmos", "decentraland", "ethereum", "matic-network", "polkadot", "solana", "the-graph", "theta-token"]



def get_fear_greed():
	'''
	Pulls the data for the fear and greed index at the time it's called.
	Returns an int.
	'''
	return int(requests.get("https://api.alternative.me/fng/?date_format=cn").json()["data"][0]["value"])



def get_fear_greed_by_range(n_days):
	return requests.get(f"https://api.alternative.me/fng/?limit={n_days}&date_format=cn").json()["data"]
	


def get_historic_data(coin, date):
	'''
	Pulls all data from coingecko for specified coin on specified date.
	Returns a dictionary.
	'''
	return requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/history?date={date}").json()



def get_correct_date_format(date):
	'''
	Puts the Python datetime into a format the coingecko api finds more copacetic i.e., dd-mm-yyyy
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



def get_community_score(data):
	'''
	Calculates a score for a coin's community health based on social media presence (Facebook, Twitter, & Reddit)
	'''
	score = 0

	for key in data.keys():
		if data[key]:
			# type cast to float is because one of the metrics was stored as a str instead of a numeric type
			score += float(data[key])

	return score



def get_dev_score(data):
	'''
	NOTE: NO LONGER IN USE BECAUSE NOT ALL COINS HAVE THIS INFO
	Calculates a score related to codebase health based on the recent activity seen by the devs
	'''
	score = 0

	if data["closed_issues"] and data["total_issues"]:
		score += (data["closed_issues"] / data["total_issues"]) 

	if data["pull_requests_merged"]: 
		score += data["pull_requests_merged"] 
	
	if data["code_additions_deletions_4_weeks"]["additions"]:
		score += data["code_additions_deletions_4_weeks"]["additions"] 

	if data["code_additions_deletions_4_weeks"]["deletions"]: 
		score += abs(data["code_additions_deletions_4_weeks"]["deletions"]) 

	if data["commit_count_4_weeks"]:
		score += data["commit_count_4_weeks"]

	return score



def get_public_interest_score(data):
	'''
	Calculates the score related to searches for the coin
	'''
	score = 0

	for key in data.keys():
		if data[key]:
			score += data[key]

	return score



def extract_basic_data(data, date):
	'''
	Extracts all useful information from the coingecko data.
	Returns a dictionary.
	'''
	data_dict = {}

	date = datetime.strptime(date, "%d-%m-%Y")
	
	data_dict["date"] = str(date.year) + "-" + str(date.month) + "-" + str(date.day)

	if "market_data" in data.keys():
		data_dict["price"] = data["market_data"]["current_price"]["twd"]
		data_dict["market_cap"] = data["market_data"]["market_cap"]["twd"]
		data_dict["volume"] = data["market_data"]["total_volume"]["twd"]
	else:
		data_dict["price"] = 0
		data_dict["market_cap"] = 0
		data_dict["volume"] = 0

	if "community_data" in data.keys():
		data_dict["community_score"] = get_community_score(data["community_data"])
	else:
		data_dict["community_score"] = 0

	if "public_interest_stats" in data:
		data_dict["public_interest_score"] = get_public_interest_score(data["public_interest_stats"])
	else:
		data_dict["public_interest_score"] = 0

	return data_dict



def get_time():
	'''
	Returns current time rounded to milliseconds
	'''
	return int(round(time.time() * 1000))



def merge_datasets(coin, list_of_datasets, all_data=False):
	'''
	Merges two or more datasets.
	
	NOTE: Param list_of_datasets must be a list of pandas DataFrames
	'''
	merged_data = pd.concat(list_of_datasets)
	merged_data["date"] = pd.to_datetime(merged_data["date"], dayfirst=True, infer_datetime_format=True)
	merged_data = merged_data.sort_values(by=["date"], ascending=False)
	# if merging all datasets into mega training dataset, more than date must be unique
	if all_data:
		subset_cols = ["date", "price", "market_cap", "vol"]
	else:
		subset_cols = ["date"]
	merged_data = merged_data.drop_duplicates(subset=subset_cols)
	merged_data = merged_data.reset_index()
	merged_data = merged_data.drop(columns=["index"])

	merged_data.to_csv(f"datasets/raw/{coin}_historical_data_raw.csv", index=False)



def merge_new_dataset_with_old(coin, by_range=True):
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



def fetch_missing_data_by_dates(coin, dates):
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
			historical_data = pd.concat(more_data)

	# save as CSV
	coin_data = pd.DataFrame(historical_data)
	coin_data.to_csv(f"datasets/raw/{coin}_historical_data_by_date.csv", index=False)
		
	print(f"{coin} data successfully pulled and stored.")
	
	return coin_data



def fetch_missing_data_by_range(coin, n_days, start_delta):
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
			historical_data = pd.concat(more_data)

	# save as CSV
	coin_data = pd.DataFrame(historical_data)
	coin_data.to_csv(f"datasets/raw/{coin}_historical_data_by_range.csv", index=False)
		
	print(f"{coin} data successfully pulled and stored.")


#fetch_missing_data_by_range("aave", 120, 109)

def run(how_far_back):
	'''
	NOTE: param how_far_back indicates how many days counting backwards from today to collect data for.
	'''
	today = date.today()
	api_calls = 0
	api_call_cycle_start = get_time() 
	fear_greed = get_fear_greed_by_range(how_far_back)
	
	for coin in ["aave"]: #coin_id:
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
		
		print("BROKE")

		# save as CSV
		coin_data = pd.DataFrame(historical_data)
		coin_data.to_csv(f"datasets/raw/{coin}_historical_data_raw.csv", index=False)
		
		# if missing dates
		if len(missing_dates) > 0:
			fetch_missing_data_by_dates(coin, missing_dates)
			merge_new_dataset_with_old(coin, by_range=False)

		print(f"{coin} data successfully pulled and stored.")



#run(600)

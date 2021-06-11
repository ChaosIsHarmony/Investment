'''
USED AS PART OF THE PROCESS TO CREATE A DATASET WITH WHICH TO TRAIN SUPERVISED LEARNING ML ALGORITHMS.

FUNCTION: TO PULL DATA FROM COINGECKO DATABASE AND STORE AS A CSV.

NOTE: THE 'SIGNAL' COLUMN IS FILLED IN BY A HUMAN IN HINDSIGHT WITH THE CORRECT ACTION GIVEN THE STATE OF THE MARKET AT THE TIME.
'''
import requests
import json
from datetime import date, datetime, timedelta
import time
import pandas as pd
import numpy as np

coin_id = ["algorand", "bitcoin", "cardano", "chainlink", "cosmos", "ethereum", "matic-network", "polkadot", "solana", "theta-token"]

API_URL = "https://api.coingecko.com/api/v3"


def get_historic_data(coin, date):
	'''
	Pulls all data from coingecko for specified coin on specified date
	'''
	return requests.get(API_URL + f"/coins/{coin}/history?date={date}").json()


def get_correct_date_format(date):
	'''
	Puts the Python datetime into a format the coingecko api finds more copacetic
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
	Extracts all useful information from the coingecko data
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

	if "community_data" in data.keys():
		data_dict["community_score"] = get_community_score(data["community_data"])
	else:
		data_dict["community_score"] = 0

	if "developer_data" in data.keys():
		data_dict["dev_score"] = get_dev_score(data["developer_data"])
	else:
		data_dict["dev_score"] = 0

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


def merge_datasets(list_of_datasets):
	'''
	Merges two or more datasets
	
	NOTE: param must be a list of pandas dataframes
	'''
	merged_data = pd.concat(list_of_datasets)
	merged_data["date"] = pd.to_datetime(merged_data["date"], dayfirst=True, infer_datetime_format=True)
	merged_data = merged_data.sort_values(by=["date"], ascending=False)
	merged_data = merged_data.reset_index()
	merged_data = merged_data.drop(columns=["Unnamed: 0", "index"])

	merged_data.to_csv(f"datasets/raw/{coin}_historical_data.csv")


def run():

	today = date.today()
	api_calls = 0
	api_call_cycle_start = get_time() 


	for coin in coin_id:
		date_delta = -1 
		error_counter = 0
		has_next = True

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
			date_delta+= 1
			next_date = get_correct_date_format(today - timedelta(date_delta))
			try:
				error_counter = 0
				data = get_historic_data(coin, next_date)
			except Exception as e:
				print(f"Error: {e}")
				print(f"Coin: {coin}")
				print(f"Date that failed: {next_date}")
				print(f"Days from today: {date_delta}")
				error_counter += 1
				if error_counter > 15:
					has_next = False
				continue

			if date_delta > 600 or "error" in data.keys():
				has_next = False
				continue	

			historical_data.append(extract_basic_data(data, next_date))
		

		# save as CSV
		coin_data = pd.DataFrame(historical_data)
		coin_data.to_csv(f"datasets/raw/{coin}_historical_data.csv")
		
		print(f"{coin} data successfully pulled and stored.")



#run()

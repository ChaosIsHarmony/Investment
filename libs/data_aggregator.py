'''
USED TO CREATE A DATASET WITH WHICH TO TRAIN SUPERVISED LEARNING ML ALGORITHMS.

NOTE: THE 'SIGNAL' COLUMN IS FILLED IN BY A HUMAN IN HINDSIGHT WITH THE CORRECT ACTION GIVEN THE STATE OF THE MARKET AT THE TIME.
'''
import requests
import json
from datetime import date, datetime, timedelta
import pandas as pd

coin_id = ["algorand"]
#coin_id = ["algorand", "bitcoin", "cardano", "chainlink", "cosmos", "ethereum", "matic-network", "polkadot", "solana", "theta-token"]

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
	data_dict["price"] = data["market_data"]["current_price"]["twd"]
	data_dict["market_cap"] = data["market_data"]["market_cap"]["twd"]
	data_dict["volume"] = data["market_data"]["total_volume"]["twd"]
	data_dict["community_score"] = get_community_score(data["community_data"])
	data_dict["dev_score"] = get_dev_score(data["developer_data"])
	data_dict["public_interest_score"] = get_public_interest_score(data["public_interest_stats"])

	return data_dict


def process_data(data):
	'''
	Processes the basic data provided by coingecko in the following ways:
		
		- Fills in missing values
		- Normalizes all values by dividing by the max value in each category
		- Calculates Simple Moving Averages for a variety of intervals
	'''
	# Fill in missing values
	# Normalize
	# Calculate SMAs (5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250)
	pass


def run():

	today = date.today()

	for coin in coin_id:
		date_delta = -1 
		has_next = True

		# Extract basic data
		historical_data = []
		while has_next:
			date_delta+=1
			if date_delta > 10:
				break
			next_date = get_correct_date_format(today - timedelta(date_delta))
			try:
				data = get_historic_data(coin, next_date)
			except Exception as e:
				print("Error: " + e)
				print(next_date)
				print(date_delta)
				has_next = False
				continue

			if "error" in data.keys():
				has_next = False
				continue	

			historical_data.append(extract_data(data, next_date))
		
		# Clean/Process data
		coin_data = pd.DataFrame(historical_data)
		coin_data = process_data(coin_data)

		# save as CSV
		coin_data.to_csv(f"{coin}_data.csv")

run()

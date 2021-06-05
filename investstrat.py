import requests
import json
import copy
from datetime import date, datetime, timedelta 
'''
Interesting API calls:
/coins/list
/coins/{id}/market_chart?vs_currency=twd&days=max&interval=monthly
'''


API_URL = "https://api.coingecko.com/api/v3"

coin_id = ["algorand", "bitcoin", "cardano", "chainlink", "ethereum", "polkadot", "solana"]


def get_price_on(coin, date):
	return requests.get(API_URL + f"/coins/{coin.get('id')}/history?date={date}").json()


def get_chart(coin, days):
	return requests.get(API_URL + f"/coins/{coin}/market_chart?vs_currency=twd&days={days}&interval=daily").json()


def find_max(prices, start=1):
	assert len(prices) > 0, "cannot process empty list"
	assert start <= len(prices), "start must not exceed price list length"

	# if subarray of length 1, return the first item
	if len(prices)-start == 0:
		return prices[0][1]

	# else compare elements
	max_price = 0
	for i in range(start, len(prices)):
		if prices[-i][1] > max_price:
			max_price = prices[-i][1]

	return max_price


def normalize(prices, start=1):
	assert len(prices) > 0, "cannot process empty list"
	assert type(prices[0][1]) == float, "cannot normalize non-numeric data"
	assert start <= len(prices), "start must not exceed price list length"

	deep_copy_prices = copy.deepcopy(prices)

	max_price = find_max(deep_copy_prices, start)
	for i in range(len(deep_copy_prices)):
		if deep_copy_prices[i][1] == max_price:
			deep_copy_prices[i][1] = 1
		else:
			deep_copy_prices[i][1] /= max_price

	return deep_copy_prices


def calc_SMA(prices, key, start=1):
	assert type(key) == str, "key should be of type str"
	assert int(key) >= 150, "key too small; must be 150 or greater"
	assert len(prices) > 0, "cannot process an empty price list"
	assert 150 <= len(prices) - start, "start must not exceed price list length - 150"

	SMA = {}
	total = 0
	for i in range(start, len(prices)+1):
		if i-start == 10:
			SMA["10"] = total/10
		elif i-start == 50:
			SMA["50"] = total/50
		elif i-start == int(key):
			SMA[key] = total/int(key)
		total += prices[-i][1]

	return SMA


def recent_risk_delta(SMA):
	assert SMA["10"] and SMA["50"], "Both 10-day MA and 50-day MA must exist"

	return SMA["10"] - SMA["50"]


def decide(risk, d_risk, cur_max_ratio):
	'''
	Heuristic for BUY/SELL/HODL based on all metrics

	params:
		risk - 50-day MA to x-day MA ratio (x provided by user; most often 350)
		d_risk - Change from 100-day MA to 50-day MA
		cur_max_ratio - Current price to All-time High price ratio
	'''
	risk 

	if d_risk > 0.05 and cur_max_ratio > 0.8:
		if risk > 2.5:
			return "SELL 3Y"
		elif risk > 2.25:
			return "SELL 2Y"
		elif risk > 1.75:
			return "SELL Y"
		elif d_risk > 0.15:
			return "SELL Y"
		else:
			return "HODL"
	elif risk < 1.25 and d_risk < -0.1 and cur_max_ratio < 0.6:
		return "BUY 2X"
	elif d_risk < -0.05 and cur_max_ratio < 0.7:
		if 1.0 <= risk < 1.5:
			return "BUY X"
		elif 0.75 <= risk < 1.0:
			return "BUY 2X"
		elif risk < 0.75:
			return "BUY 3X"
		else:
			return "HODL"
	
	return "HODL"

def generate_report(SMA, prices, key="350", start=1):
	risk = SMA["50"]/SMA[key]
	d_risk = recent_risk_delta(SMA)
	cur_max_ratio = prices[-start][1] / find_max(prices, start)
	decision = decide(risk, d_risk, cur_max_ratio)

	return risk, d_risk, cur_max_ratio, decision
	

def print_report(SMA, prices, key="350", start=1, show_all=True):
	if key in SMA.keys():
		risk, d_risk, cur_max_ratio, decision = generate_report(SMA, prices, key, start)
		
		if show_all or decision == "BUY" or decision == "SELL":
			print(f"\tRisk 50/{key}: ", end="")
			print(f"{risk:>16.4f}")
			print("\t10 - 50 MA delta: ", end="")
			print(f"{d_risk:>11.4f}")
			print(f"\t\t10 SMA: {SMA['10']:.4f} | 50 SMA: {SMA['50']:.4f}" )
			print("\tCur/Max Price: ", end="")
			print(f"{cur_max_ratio:>14.4f}")
			print()
			print("DECISION:", decision)
			return decision

def most_recent_report(coin, key, days):
	prices = get_chart(coin, days+1)["prices"]
	SMA = {}

	print(coin)
	print("Present Day")
	prices = normalize(prices, 1)
	SMA = calc_SMA(prices, key, 1)
	print_report(SMA, prices, key, 1)
	print("------------")
	print()
	print("------------")


def in_depth_report(coin, key, days):
	prices_orig = get_chart(coin, days+1)["prices"]
	SMA = {}
	signal_changes = {}	
	prev_signal = ""
	today = date.today()

	print(coin)
	print(today)
	prices_norm = normalize(prices_orig, 1)
	SMA = calc_SMA(prices_norm, key, 1)
	print_report(SMA, prices_norm, key, 1)
	print("------------")
	print()
	print("------------")
	
	for i in range(2, days-int(key)):
		print(today - timedelta(i))
		prices_norm = normalize(prices_orig, i)
		SMA = calc_SMA(prices_norm, key, i)

		# not enough data left to perform analysis
		if key not in SMA.keys():
			break

		decision = print_report(SMA, prices_norm, key, i)
		if decision != prev_signal:
			risk = SMA["50"]/SMA[key]
			d_risk = recent_risk_delta(SMA)
			cur_max_ratio = prices_norm[-i][1] / find_max(prices_norm, i)

			signal_changes[i] = [decision, (risk+d_risk+cur_max_ratio)]
			prev_signal = decision
		print("------------")
		print()
		print("------------")
	
	return signal_changes


def run():
	while (True):
		print("Available coins: ", coin_id)
		coin = input("Which coin? ")

		# not an option
		if coin not in coin_id and coin != "all":
			break
 
		key = input("What MA should the 50-day MA be compared to? (increments of 50) ")
		days = int(input("How far back? (in days) "))
	
		# Report on all coins
		if coin == "all":
			for c in coin_id:
				most_recent_report(c, key, days)
			continue

		# In-depth report on single coin
		in_depth_report(coin, key, days)


# Uncomment if running in CLI
# Comment if running in Python Shell
#run()

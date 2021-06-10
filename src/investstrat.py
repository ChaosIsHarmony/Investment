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

coin_id = ["algorand", "bitcoin", "cardano", "chainlink", "cosmos", "ethereum", "matic-network", "polkadot", "solana", "theta-token"]



def get_price_on(coin, date):
	'''
	Gets the price of the specified coin on the specified date.
	
	NOTE: param date requires "dd-mm-yyyy" format as per the coingecko api requirements.
	'''
	return requests.get(API_URL + f"/coins/{coin}/history?date={date}").json()



def get_chart(coin, days):
	'''
	Gets the market chart for the specified coin for a specified number of days extending backwards from the present.

	NOTE: if days == "max", then it will retrieve all possible data for the given coin.
	'''
	return requests.get(API_URL + f"/coins/{coin}/market_chart?vs_currency=twd&days={days}&interval=daily").json()



def find_max(prices, start=1):
	'''
	Finds highest price starting from the index len(prices)-start; thus, if start == 1, it starts from the last index (i.e., the most recent date).

	NOTE: this method searches backwards as the most recent data is listed last (i.e., the first item in the array is the oldest; the last, the most recent).
	'''
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
	'''
	Normalizes prices by finding highest price in the given range and then dividing all by that number.
	'''
	assert len(prices) > 0, "cannot process empty list"
	assert type(prices[0][1]) == float, "cannot normalize non-numeric data"
	assert start <= len(prices), "start must not exceed price list length"

	deep_copy_prices = copy.deepcopy(prices)

	max_price = find_max(deep_copy_prices, start)
	for i in range(len(deep_copy_prices)):
		deep_copy_prices[i][1] /= max_price

	return deep_copy_prices



def calc_SMA(prices, key, start=1):
	'''
	Calculates the Simple Moving Average for three intervals:
		- 10-day
		- 50-day
		- key-day (where the key param is >= 100)

	NOTE: the start param is used to indicate the starting date (i.e., x days backward from the present day)
	'''
	assert type(key) == str, "key should be of type str"
	assert int(key) >= 100, "key too small; must be 100 or greater"
	assert len(prices) > 0, "cannot process an empty price list"
	assert 100 <= len(prices) - start, "start must not exceed price list length - 100"

	SMA = {}
	total = 0
	for i in range(start, len(prices)+1):
		if i-start == 5:
			SMA["5"] = total/5
		elif i-start == 10:
			SMA["10"] = total/10
		elif i-start == 50:
			SMA["50"] = total/50
		elif i-start == 100:
			SMA["100"] = total/100
		elif i-start == int(key):
			SMA[key] = total/int(key)
		total += prices[-i][1]

	return SMA



def recent_risk_delta(SMA):
	'''
	Calculates the difference between the 10-day moving average and the 50-day moving average.

	NOTE: this is a upward/downward + magnitude trend indicator: higher, positive numbers means rapid growth in short time; lower, negative numbers mean rapid loss in short time; whereas near-zero numbers mean
	'''
	assert SMA["5"] and SMA["10"] and SMA["50"] and SMA["100"], "5-day, 10-day, 50-day, and 100-day MA must exist"

	return SMA["5"] - SMA["10"], SMA["10"] - SMA["50"], SMA["50"] - SMA["100"]



def decide(risk, d_risk_5, d_risk_10, d_risk_50, cur_max_ratio):
	'''
	Heuristic for BUY/SELL/HODL based on all metrics

	params:
		risk - 50-day MA to x-day MA ratio (x provided by user; most often 200, as is indicator of golden cross or death cross)
		d_risk - Changes between 5, 10, 50, and 100-day MAs
		cur_max_ratio - Current price to All-time High price ratio
	'''

	if d_risk_10 > 0.05 and cur_max_ratio > 0.8:
		if risk > 2.5:
			return "SELL 3Y"
		elif risk > 2.25:
			return "SELL 2Y"
		elif risk > 1.75:
			return "SELL Y"
		elif d_risk_10 > 0.15:
			return "SELL Y"
		else:
			return "HODL"
	elif risk < 1.25 and d_risk_10 < -0.1 and cur_max_ratio < 0.6:
		return "BUY 2X"
	elif d_risk_10 < -0.05 and cur_max_ratio < 0.7:
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
	d_risk_5, d_risk_10, d_risk_50 = recent_risk_delta(SMA)
	cur_max_ratio = prices[-start][1] / find_max(prices, start)
	decision = decide(risk, d_risk_5, d_risk_10, d_risk_50, cur_max_ratio)

	return risk, d_risk_5, d_risk_10, d_risk_50, cur_max_ratio, decision
	


def print_report(SMA, prices, key="350", start=1, show_all=True):
	if key in SMA.keys():
		risk, d_risk_5, d_risk_10, d_risk_50, cur_max_ratio, decision = generate_report(SMA, prices, key, start)
		
		if show_all or decision == "BUY" or decision == "SELL":
			print(f"\tRisk 50/{key}: ", end="")
			print(f"{risk:>16.4f}")
			print("\tMA Deltas:")
			print(f"\t\t5 SMA: {SMA['5']:.4f} | 10 SMA: {SMA['10']:.4f} | 50 SMA: {SMA['50']:.4f} | 100 SMA: {SMA['100']:.4f}" )
			print("\t\t5 - 10 MA delta: ", end="")
			print(f"{d_risk_5:>12.4f}")
			print("\t\t10 - 50 MA delta: ", end="")
			print(f"{d_risk_10:>11.4f}")
			print("\t\t50 - 100 MA delta: ", end="")
			print(f"{d_risk_50:>11.4f}")
			print("\tCur/Max Price: ", end="")
			print(f"{cur_max_ratio:>14.4f}")
			print()
			print("DECISION:", decision)
			return decision



def most_recent_report(coin, key, days):
	'''
	Only provides a report for the present day's risk and investment decision
	'''
	prices = get_chart(coin, days+1)["prices"]
	SMA = {}
	today = date.today()

	print(coin)
	print(today)
	prices = normalize(prices, 1)
	SMA = calc_SMA(prices, key, 1)
	print_report(SMA, prices, key, 1)
	print("------------")
	print()
	print("------------")



def in_depth_report(coin, key, days):
	'''
	Gives metric and decision based on data up to a given point in history for a specified interval.

	NOTE: key param is the x-day moving average (e.g., 200, 350, etc.) with which a risk ratio will be calculated using the 50-day moving average.
	'''
	prices_orig = get_chart(coin, days+1)["prices"]
	SMA = {}
	signal_changes = {}	
	prev_signal = ""
	today = date.today()

	print(coin)
	print(today)
	prices_norm = normalize(prices_orig, 1)
	SMA = calc_SMA(prices_norm, key, 1)
	prev_signal = print_report(SMA, prices_norm, key, 1)
	print("------------")
	print()
	print("------------")

	# calculates historical decisions based only on historical data
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
			d_risk_5, d_risk_10, d_risk_50 = recent_risk_delta(SMA)
			cur_max_ratio = prices_norm[-i][1] / find_max(prices_norm, i)

			signal_changes[i] = [decision, risk, d_risk_5, d_risk_10, d_risk_50, cur_max_ratio]
			prev_signal = decision
		print("------------")
		print()
		print("------------")
	
	return signal_changes

import requests
import json
 
'''
Interesting API calls:
/coins/list
/coins/{id}/market_chart?vs_currency=twd&days=max&interval=monthly

UNUSED
x = requests.get(API_URL + "/coins/list?include_platform=false")
for coin in x.json():
	coin_name = coin.get("symbol")
	for coin_sym in coin_symbol_list_hodl:
		if coin_sym in coin_name:
			date = "30-12-2020"
			# history for given date
			print(coin_name, get_price_on(coin, date).keys())
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

	max_price = find_max(prices, start)
	for i in range(len(prices)):
		if prices[i][1] == max_price:
			prices[i][1] = 1
		else:
			prices[i][1] /= max_price

	return prices


def calc_SMA(prices, key, start=1):
	assert int(key) >= 150, "key too small; must be 150 or greater"
	assert len(prices) > 0, "cannot process an empty price list"
	assert start <= len(prices)-150, "start must not exceed price list length - 150"

	SMA = {}
	total = 0
	for i in range(start, len(prices)+1):
		if i-start == 50:
			SMA["50"] = total/50
		elif i-start == 100:
			SMA["100"] = total/100
		elif i-start == int(key):
			SMA[key] = total/int(key)
		total += prices[-i][1]

	return SMA


def recent_risk_delta(SMA):
	assert SMA["50"] and SMA["100"], "Both 50-day MA and 100-day MA must exist"

	return SMA["50"] - SMA["100"]


def decide(risk, d_risk, cur_max_ratio):
	'''
	Heuristic for BUY/SELL/HODL based on all metrics

	params:
		risk - 50-day MA to x-day MA ratio (x provided by user; most often 350)
		d_risk - Change from 100-day MA to 50-day MA
		cur_max_ratio - Current price to All-time High price ratio
	'''
	tot_risk = risk + d_risk + cur_max_ratio
	print(f"Total risk aggregate:  {tot_risk:.4f}")

	if tot_risk > 5:
		return "SELL 3Y"
	elif tot_risk > 4.5:
		return "SELL 2Y"
	elif tot_risk > 4:
		return "SELL Y"
	elif tot_risk > 3.5:
		return "HODL"
	elif tot_risk > 2.5:
		return "BUY X"
	elif tot_risk > 2:
		return "BUY 2X"
	else:
		return "BUY 3x"
	

def print_report(SMA, prices, key="350", start=1, show_all=True):
	if key in SMA.keys():
		risk = SMA["50"]/SMA[key]
		d_risk = recent_risk_delta(SMA)
		cur_max_ratio = prices[-start][1] / find_max(prices, start)
		decision = decide(risk, d_risk, cur_max_ratio)
	
		if show_all or decision == "BUY" or decision == "SELL":
			print(f"\tRisk 50/{key}: ", end="")
			print(f"{risk:>16.4f}")
			print("\t50 - 100 MA delta: ", end="")
			print(f"{d_risk:>10.4f}")
			print(f"\t\t50 SMA: {SMA['50']:.4f} | 100 SMA: {SMA['100']:.4f}" )
			print("\tCur/Max Price: ", end="")
			print(f"{cur_max_ratio:>14.4f}")
			print()
			if int(key) != 350:
				print("WARNING: Decision is calibrated for 50/350 MA comparison")
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

	print(coin)
	print("Present Day")
	prices_norm = normalize(prices_orig, 1)
	SMA = calc_SMA(prices_norm, key, 1)
	print_report(SMA, prices_norm, key, 1)

	for i in range(1, days-int(key)):
		print(f"{i} days ago")
		prices_norm = normalize(prices_orig, i)
		SMA = calc_SMA(prices_norm, key, i)

		# not enough data left to perform analysis
		if key not in SMA.keys():
			break

		decision = print_report(SMA, prices_norm, key, i)
		if decision != prev_signal:
			signal_changes[i] = [decision, SMA["50"]/SMA[key]]
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

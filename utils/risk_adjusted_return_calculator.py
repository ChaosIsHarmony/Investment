import pandas as pd
import numpy as np
import common
import datetime


def calculate_excess_return(data, interval, verbose=False):
	'''
	Excess return = closing_price - starting_price
	'''
	risk_free_rate = 0.005

	# calculate daily price change
	daily_price_delta = []
	for r in range(0, interval):
		daily_price_delta.append(((data.iloc[r, 1] - data.iloc[r+1, 1])/data.iloc[r-1, 1]))
	
	if verbose:
		neg = [x for x in daily_price_delta if x < 0]
		pos = [x for x in daily_price_delta if x >= 0]
		print(f"Neg: {len(neg)} | {sum(neg)}")
		print(f"Pos: {len(pos)} | {sum(pos)}")

	# calculate sharpe ratio components
	starting_price = data.iloc[interval-1, 1]
	closing_price = data.iloc[0, 1]
	asset_performance_rate = ((closing_price - starting_price) / starting_price)
	excess_return = asset_performance_rate - risk_free_rate

	return excess_return, daily_price_delta



def calculate_ulcer_index(data, interval):
	# R_i_sq = ((price_i / max_price) - 1) ** 2
	curr_row = interval
	max_price = data.iloc[curr_row, 1]
	sum_R_sq = 0
	while curr_row > 1:
		curr_row -= 1
		new_price = data.iloc[curr_row, 1] 
		if new_price > max_price:
			max_price = new_price
		else:
			sum_R_sq += ((new_price / max_price) - 1) ** 2

	ulcer_index = np.sqrt(sum_R_sq / interval)

	return ulcer_index
    


def calculate_ulcer_performance_index(data, interval):
	excess_return, _ = calculate_excess_return(data, interval)	
	volatility = calculate_ulcer_index(data, interval)

	upi = excess_return / volatility

	return upi



def calculate_sharpe_ratio(data, interval):
	excess_return, daily_price_delta = calculate_excess_return(data, interval)
	volatility = np.std(daily_price_delta)*np.sqrt(interval) # multiplying by np.sqrt(interval) scales std to size of the interval

	sharpe_ratio = excess_return / volatility

	return sharpe_ratio



def get_current_sharpe_ratio(data):
	return calculate_sharpe_ratio(data, 365)



def get_custom_sharpe_ratio(data, interval):
	return calculate_sharpe_ratio(data, interval)



def prepare_dataframe(coin):
	data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")

	# drop irrelevant columns
	data = data.drop(columns=["market_cap", "volume", "fear_greed"])
	last_ind = data.shape[0] - 1
	data = common.handle_missing_data(data, data.iloc[0,0], data.iloc[last_ind,0])

	return data



def get_sharpe_ratio(coin):
	'''
	If possible, returns the 365-day Sharpe Ratio
	Else, returns the max interval Sharpe Ratio.
	'''
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	try:
		return get_current_sharpe_ratio(data)
	except:
		return get_custom_sharpe_ratio(data, interval)



def get_upi(coin):
	'''
	If possible, returns the 365-day UPI
	Else, returns the max interval UPI.
	'''
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	try:
		return calculate_ulcer_performance_index(data, 365)
	except:
		return calculate_ulcer_performance_index(data, interval)



def print_sharpe_ratio(coin):
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	try:
		print(f"Current yearly Sharpe ratio for {coin}: {get_current_sharpe_ratio(data):.6f}")
	except:
		print(f"Custom Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, interval):.6f}")

	

def print_ulcer_performance_index(coin):
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	try:
		print(f"Current yearly UPI for {coin}: {calculate_ulcer_performance_index(data, 365):.6f}")
	except:
		print(f"Custom UPI for {coin} of {interval} days: {calculate_ulcer_performance_index(data, interval):.6f}")



def print_personal_sharpe_ratio(coin, interval):
	data = prepare_dataframe(coin)
	print(f"Personal Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, interval):.6f}")
	


def print_personal_ulcer_performance_index(coin, interval):
	data = prepare_dataframe(coin)
	print(f"Personal UPI for {coin} of {interval} days: {calculate_ulcer_performance_index(data, interval):.6f}")



def print_max_sharpe_ratio(coin):
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	print(f"Max Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, interval):.6f}")
	


def print_max_ulcer_performance_index(coin):
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	print(f"Max UPI for {coin} of {interval} days: {calculate_ulcer_performance_index(data, interval):.6f}")
	


def print_custom_sharpe_ratio(coin, interval):
	data = prepare_dataframe(coin)
	try:
		print(f"Custom Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, interval):.6f}")
	except:
		interval = data.shape[0] - 1
		print(f"Custom Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, interval):.6f}")
	


def print_custom_ulcer_performance_index(coin, interval):
	data = prepare_dataframe(coin)
	try:
		print(f"Custom UPI ratio for {coin} of {interval} days: {calculate_ulcer_performance_index(data, interval):.6f}")
	except:
		interval = data.shape[0] - 1
		print(f"Custom UPI ratio for {coin} of {interval} days: {calculate_ulcer_performance_index(data, interval):.6f}")
	



if __name__ == "__main__":
	while(True):
		choice = input("Which version?\n(0) Cancel\n(1) Print Yearly\n(2) Print Personal\n(3) Print Max\n(4) Custom\nChoice: ")
		if choice == "0" or choice == "1" or choice == "2" or choice == "3":
			break
		elif choice == "4":
			while True:
				try:
					interval = int(input("What time interval (in days)? [e.g., 50] "))
					break
				except:
					print("Must be an integer value.")
			break
			

	if choice != "0":
		for coin in common.coins:
			if choice == "1":
				print_sharpe_ratio(coin)
				print_ulcer_performance_index(coin)
				print()
			elif choice == "2":
				today = datetime.date.today()
				beginning = datetime.date(2021, 4, 1)
				diff = today - beginning
				interval = diff.days
				print_personal_sharpe_ratio(coin, interval)
				print_personal_ulcer_performance_index(coin, interval)
				print()
			elif choice == "3":
				print_max_sharpe_ratio(coin)
				print_max_ulcer_performance_index(coin)
				print()
			else:
				print_custom_sharpe_ratio(coin, interval)
				print_custom_ulcer_performance_index(coin, interval)
				print()

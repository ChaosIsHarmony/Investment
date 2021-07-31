import pandas as pd
import data_preprocessor as dpp
import ulcer_index_calculator as uic
import numpy as np
import common


def calculate_ulcer_performance_index(data, interval):
	risk_free_rate = 0.005

	# calculate daily price change
	daily_price_delta = []
	for r in range(0, interval):
		daily_price_delta.append(((data.iloc[r, 1] - data.iloc[r+1, 1])/data.iloc[r-1, 1]))
	
	# calculate sharpe ratio components
	excess_return = sum(daily_price_delta) - risk_free_rate
	volatility = uic.calculate_ulcer_index(data, interval)

	upi = excess_return / volatility

	return upi



def calculate_sharpe_ratio(data, interval):
	'''
	Calculates the Sharpe ratio for a given interval.
	NOTE: Assumes a risk free rate of 0.5% which is about 5x the rate of a 1-yr US treasury bond as of 07/21.
	'''
	risk_free_rate = 0.005

	# calculate daily price change
	daily_price_delta = []
	for r in range(0, interval):
		daily_price_delta.append(((data.iloc[r, 1] - data.iloc[r+1, 1])/data.iloc[r-1, 1]))
	
	# calculate sharpe ratio components
	excess_return = sum(daily_price_delta) - risk_free_rate
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
	data = dpp.handle_missing_data(data, data.iloc[0,0], data.iloc[last_ind,0])
	return data



def get_sharpe_ratio(coin):
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	try:
		return get_current_sharpe_ratio(data, coin)
	except:
		return get_custom_sharpe_ratio(data, coin, interval)



def get_upi(coin):
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


def print_personal_sharpe_ratio(coin):
	data = prepare_dataframe(coin)
	interval = 17*7
	print(f"Personal Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, coin, interval):.6f}")
	


def print_max_sharpe_ratio(coin):
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	print(f"Max Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, coin, interval):.6f}")
	

if __name__ == "__main__":
	while(True):
		choice = input("Which version?\n(0) Cancel\n(1) Print Yearly\n(2) Print Personal\n(3) Print Max\nChoice: ")
		if choice == "0" or choice == "1" or choice == "2" or choice == "3":
			break


	if choice != "0":
		for coin in common.coins:
			if choice == "1":
				print_sharpe_ratio(coin)
				print_ulcer_performance_index(coin)
			elif choice == "2":
				print_personal_sharpe_ratio(coin)
			else:
				print_max_sharpe_ratio(coin)

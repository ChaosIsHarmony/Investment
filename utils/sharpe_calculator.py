import pandas as pd
import data_preprocessor as dpp
import numpy as np
import common

def calculate_sharpe_ratio(data, coin, interval):
	'''
	Calculates the Sharpe ratio for a given interval.
	NOTE: Assumes a risk free rate of 0.5% which is about 5x the rate of a 1-yr US treasury bond.
	'''
	risk_free_rate = 0.005

	# calculate daily price change
	daily_price_delta = []
	for r in range(0, interval):
		daily_price_delta.append(((data.iloc[r, 1] - data.iloc[r+1, 1])/data.iloc[r-1, 1]))
	
	# calculate sharpe ratio components
	intervallic_excess_return = sum(daily_price_delta) - risk_free_rate
	intervallic_volatility = np.std(daily_price_delta)*np.sqrt(interval) # multiplying by np.sqrt(interval) scales std to size of the interval
	
	sharpe_ratio = intervallic_excess_return / intervallic_volatility

	return sharpe_ratio



def get_current_sharpe_ratio(data, coin):
	return calculate_sharpe_ratio(data, coin, 365)



def get_custom_sharpe_ratio(data, coin, interval):
	return calculate_sharpe_ratio(data, coin, interval)



def prepare_dataframe(coin):
	data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")

	# drop irrelevant columns
	data = data.drop(columns=["market_cap", "volume", "fear_greed"])
	last_ind = data.shape[0] - 1
	data = dpp.handle_missing_data(data, data.iloc[0,0], data.iloc[last_ind,0])
	return data



def print_sharpe_ratio(coin):
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	try:
		print(f"Current yearly Sharpe ratio for {coin}: {get_current_sharpe_ratio(data, coin)}")
	except:
		print(f"Custom Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, coin, interval)}")

	

def print_personal_sharpe_ratio(coin):
	data = prepare_dataframe(coin)
	interval = 17*7
	print(f"Personal Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, coin, interval)}")
	


def print_max_sharpe_ratio(coin):
	data = prepare_dataframe(coin)
	interval = data.shape[0] - 1
	print(f"Max Sharpe ratio for {coin} of {interval} days: {get_custom_sharpe_ratio(data, coin, interval)}")
	

if __name__ == "__main__":
	for coin in common.coins:
		#print_sharpe_ratio(coin)
		#print_personal_sharpe_ratio(coin)
		print_max_sharpe_ratio(coin)

import pandas as pd
import data_preprocessor as dpp
import numpy as np
import common

def calculate_sharpe_ratio(data, coin, days_before_today=0):
	# calculate daily price change
	last_ind = data.shape[0] - 1 - days_before_today
	daily_price_delta = [np.nan]
	for r in range(1, data.shape[0]):
		daily_price_delta.append((data.iloc[r, 1] - data.iloc[r-1, 1])/data.iloc[r-1, 1])
	data["daily_price_delta"] = daily_price_delta 
	mean_delta = np.mean(data.daily_price_delta)
	std_delta = np.std(data.daily_price_delta)


	compound_annual_growth_rate = (data.iloc[last_ind, 1] / data.iloc[0, 1]) - 1
	risk_free_rate = 0.005 # means putting it in a guaranteed interest investment, such as bonds or savings accounts
	annualized_volatility = std_delta*np.sqrt(365) # the std of daily changes multiplied by avg days traded in a year

	
	sharpe_ratio = (compound_annual_growth_rate - risk_free_rate) / annualized_volatility

	return sharpe_ratio



def get_current_sharpe_ratio(data, coin):
	return calculate_sharpe_ratio(data, coin)



def get_average_sharpe_ratio(data, coin, interval=60):
	ratios = []
	for i in range(interval):
		ratios.append(calculate_sharpe_ratio(data, coin, i))
	return sum(ratios)/len(ratios)



def prepare_dataframe(coin):
	data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")

	# drop irrelevant columns
	data = data.drop(columns=["market_cap", "volume", "fear_greed"])
	last_ind = data.shape[0] - 1
	data = dpp.handle_missing_data(data, data.iloc[0,0], data.iloc[last_ind,0])
	# reverse
	data = data.iloc[::-1]

	return data



if __name__ == "__main__":
	interval = 180
	for coin in common.coins:
		print(f"Current Sharpe ratio for {coin}: {get_current_sharpe_ratio(prepare_dataframe(coin), coin)}")
		print(f"Average Sharpe ratio for {coin} [interval={interval}]: {get_average_sharpe_ratio(prepare_dataframe(coin), coin, interval)}")

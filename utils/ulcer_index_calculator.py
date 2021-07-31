import pandas as pd
import data_preprocessor as dpp
import numpy as np

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


def prepare_dataframe(coin):
	data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")

	# drop irrelevant columns
	data = data.drop(columns=["market_cap", "volume", "fear_greed"])
	last_ind = data.shape[0] - 1
	data = dpp.handle_missing_data(data, data.iloc[0,0], data.iloc[last_ind,0])

	return data




if __name__ == "__main__":
	data = prepare_dataframe("bitcoin")
	calculate_ulcer_index(data, 365)

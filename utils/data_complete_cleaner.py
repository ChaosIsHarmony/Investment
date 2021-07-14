'''
Takes a clean dataset and prunes it so that it's suitable for training.
'''
import pandas as pd
import common

coins = ["algorand", "bitcoin", "cardano", "chainlink", "ethereum", "polkadot", "solana"]

for coin in coins: 
	data = pd.read_csv(f"datasets/clean/{coin}_historical_data_clean.csv")

	# find first instance of real SMA_350 value
	data = data.iloc[1: , :] # drop first row as it will always be normalized to 1
	start_ind = data[data["price_350_SMA"] == 1].first_valid_index()
	if start_ind != None:
		data = data.iloc[start_ind: , :]

		# find last instance of signal value and trim dataset
		data = data.iloc[:len(data)-common.SIGNAL_FOR_N_DAYS_FROM_NOW, :]
		data.to_csv(f"datasets/complete/{coin}_historical_data_complete.csv", index=False, float_format="%f")
	else:
		print(f"{coin} does not have enough data.")


import pandas as pd


coin_id = ["algorand", "bitcoin", "cardano", "chainlink", "ethereum", "polkadot", "solana"]
data_type = "raw"


for coin in coin_id:
	data = pd.read_csv(f"datasets/{data_type}/{coin}_historical_data_{data_type}.csv")

	count = 0
	for c in range(1, len(data.columns)):
		for r in range(1, len(data)-1):
			former_cell = data.iloc[r-1, c]
			curr_cell = data.iloc[r, c]
			latter_cell = data.iloc[r+1, c]
			if (curr_cell / 10) > former_cell or (curr_cell / 10) > latter_cell:
				factor = curr_cell // min(former_cell, latter_cell)
				print(c,r, factor)
				if factor >= 100:
					data.iloc[r, c] = data.iloc[r, c] / 1000
				elif factor >= 10:
					data.iloc[r, c] = data.iloc[r, c] / 100
				count += 1

	print(count)
	
	data.to_csv(f"datasets/{data_type}/{coin}_historical_data_{data_type}.csv", index=False, float_format="%f")

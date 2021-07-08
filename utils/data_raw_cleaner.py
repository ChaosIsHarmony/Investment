import pandas as pd


coin_id = ["algorand", "bitcoin", "cardano", "chainlink", "ethereum", "polkadot", "solana"]

for coin in coin_id:
	data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")

	count = 0
	for c in range(1, len(data.columns)):
		for r in range(1, len(data)-1):
			former_cell = data.iloc[r-1, c]
			curr_cell = data.iloc[r, c]
			latter_cell = data.iloc[r+1, c]
			if (curr_cell / 10) > former_cell and (curr_cell / 10) > latter_cell:
				factor = curr_cell // ((former_cell + latter_cell) / 2)
				print(c,r, factor)
				if factor >= 100:
					data.iloc[r, c] = data.iloc[r, c] / 1000
				elif factor >= 10:
					data.iloc[r, c] = data.iloc[r, c] / 100
				count += 1

	print(count)
	
	data.to_csv(f"datasets/raw/{coin}_historical_data_raw.csv", index=False, float_format="%f")

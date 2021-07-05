import pandas as pd

data = pd.read_csv("datasets/complete/bitcoin_historical_data_complete.csv")

for c in range(1, len(data.columns)-1):
	for r in range(len(data)):
		if (data.iloc[r, c] > 1) == True:
			print(f"ERROR: Unnormalized data still present in the dataset. Column: {data.columns[c]} | Row: {r} | Value: {data.iloc[r, c]}")
			data.iloc[r, c] = data.iloc[r, c] / 1000


data.to_csv("datasets/complete/bitcoin_historical_data_complete.csv", index=False)

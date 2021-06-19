import torch
import data_aggregator as dt_agg
import data_preprocessor as dt_prepro
import neural_nets as nn
import pandas as pd
from datetime import date


DECISIONS = ["BUY 2X", "BUY X", "HODL", "SELL Y", "SELL 2Y"]


def fetch_new_data(n_days):
	for coin in dt_agg.coin_id:
		dt_agg.fetch_missing_data_by_range(coin, n_days, 0)
		print(f"Successfully fetched new {coin} data")
		dt_agg.merge_new_dataset_with_old(coin)
		print(f"Successfully merged new and old {coin} data")



def process_new_data():
	start_date = str(date.today())
	end_date = "2020-08-23" #the first day of data of the youngest asset
	for coin in dt_agg.coin_id:
		print(coin)

		data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")
		data = dt_prepro.process_data(data, start_date, end_date)
		data.to_csv(f"datasets/clean/{coin}_historical_data_clean.csv", index=False)



def generate_signals():
	with open("reports/best_performers.txt") as f:
		best = f.read().splitlines() 

	coin = "bitcoin"
	for coin in dt_agg.coin_id:
		data = pd.read_csv(f"datasets/clean/{coin}_historical_data_clean.csv")
		# extracts the most recent data as a python list
		data = data[data["date"] == str(date.today())].values.tolist()[0][1:]
		n_votes = [0, 0, 0, 0, 0] # buy 2x, buy x, etc.
		
		models = {}
		for i in range(3):
			if "Laptop_0" in best[i]:
				models[best[i]] = nn.CryptoSoothsayer_Laptop_0(nn.N_FEATURES, nn.N_SIGNALS)
			elif "Laptop_1" in best[i]:
				models[best[i]] = nn.CryptoSoothsayer_Laptop_1(nn.N_FEATURES, nn.N_SIGNALS)
			elif "Laptop_2" in best[i]:
				models[best[i]] = nn.CryptoSoothsayer_Laptop_2(nn.N_FEATURES, nn.N_SIGNALS)
			elif "Pi_0" in best[i]:
				models[best[i]] = nn.CryptoSoothsayer_Pi_0(nn.N_FEATURES, nn.N_SIGNALS)
			elif "Pi_1" in best[i]:
				models[best[i]] = nn.CryptoSoothsayer_Pi_1(nn.N_FEATURES, nn.N_SIGNALS)
			elif "PC_0" in best[i]:
				models[best[i]] = nn.CryptoSoothsayer_PC_0(nn.N_FEATURES, nn.N_SIGNALS)
			elif "PC_1" in best[i]:
				models[best[i]] = nn.CryptoSoothsayer_PC_1(nn.N_FEATURES, nn.N_SIGNALS)


		for filepath in models.keys():
			model = models[filepath]
			model.load_state_dict(torch.load(filepath))
			model.to(torch.device("cpu"))
			model.eval()
			feature_tensor = torch.tensor([data], dtype=torch.float32)

			with torch.no_grad():
				output = model(feature_tensor)

			for i in range(len(n_votes)):
				n_votes[i] += float(output[0][i])


		# find intersection of answers
		'''
		If a majority suggest to BUY 2X, then BUY 2X.
		If a majority suggest to either BUY 2X or BUY X, then BUY X.
		Above likewise applies to SELL, and HODL signals.
		If split (e.g., SELL Y, BUY X, and HODL), then HODL.
		'''
		n_votes = torch.tensor(n_votes, dtype=torch.float32)
		signal = DECISIONS[torch.argmax(n_votes)]
		print(f"Signal for {coin}: {signal}")

#fetch_new_data(1)
#process_new_data()
generate_signals()

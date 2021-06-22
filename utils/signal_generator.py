import torch
import data_aggregator as dt_agg
import data_preprocessor as dt_prepro
import neural_nets as nn
import pandas as pd
from datetime import date, timedelta


DECISIONS = ["BUY 2X", "BUY X", "HODL", "SELL Y", "SELL 2Y"]


def fetch_new_data(n_days):
	for coin in dt_agg.coin_id:
		dt_agg.fetch_missing_data_by_range(coin, n_days, 0)
		print(f"Successfully fetched new {coin} data")
		dt_agg.merge_new_dataset_with_old(coin)
		print(f"Successfully merged new and old {coin} data")
		print()



def process_new_data():
	start_date = str(date.today())
	end_date = "2020-08-23" #the first day of data of the youngest asset: polkadot
	for coin in dt_agg.coin_id:
		print(coin)

		data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")
		data = dt_prepro.process_data(data, start_date, end_date)
		data.to_csv(f"datasets/clean/{coin}_historical_data_clean.csv", index=False)



def get_fg_indicator(fg_index):
	if fg_index < 0.2:
		return "Extreme Fear"
	elif fg_index < 0.4:
		return "Fear"
	elif fg_index < 0.6:
		return "Neutral"
	elif fg_index < 0.8:
		return "Greed"
	else:
		return "Extreme Greed"



def populate_stat_report(coin, data, report):
	basic_stats = ["\n\n\n", 
					f"Report for {coin.upper()}:", 
					"Basic Stats", 
					"[1.0 is the highest; 0.0 is the lowest]", 
					f"price:\t\t{data[0]:.6f}", 
					f"market_cap:\t{data[1]:.6f}", 
					f"volume:\t\t{data[2]:.6f}", 
					f"fear/greed\t{data[6]:.6f} [{get_fg_indicator(data[6])}]"]
	
	price_ratios = ["\nPrice Ratios", 
					"[>0 means greater risk/overvalued; <0 means less risk/undervalued]", 
					f"5-day/10-day:\t{data[7]/data[8]:>9.6f}", 
					f"10-day/25-day:\t{data[8]/data[9]:>9.6f}", 
					f"25-day/50-day:\t{data[9]/data[10]:>9.6f}"]

	if data[14] > 0:
		price_ratios.append("x-day/200-day")
		price_ratios.append(f"\t5-day/200-day:\t\t{data[7]/data[14]:>9.6f}")
		price_ratios.append(f"\t10-day/200-day:\t\t{data[8]/data[14]:>9.6f}")
		price_ratios.append(f"\t25-day/200-day:\t\t{data[9]/data[14]:>9.6f}")
		price_ratios.append(f"\t50-day/200-day:\t\t{data[10]/data[14]:>9.6f}")
		price_ratios.append(f"\t75-day/200-day:\t\t{data[11]/data[14]:>9.6f}")
		price_ratios.append(f"\t100-day/200-day:\t{data[12]/data[14]:>9.6f}")
	if data[15] > 0:
		price_ratios.append("x-day/250-day")
		price_ratios.append(f"\t5-day/250-day:\t\t{data[7]/data[15]:>9.6f}")
		price_ratios.append(f"\t10-day/250-day:\t\t{data[8]/data[15]:>9.6f}")
		price_ratios.append(f"\t25-day/250-day:\t\t{data[9]/data[15]:>9.6f}")
		price_ratios.append(f"\t50-day/250-day:\t\t{data[10]/data[15]:>9.6f}")
		price_ratios.append(f"\t75-day/250-day:\t\t{data[11]/data[15]:>9.6f}")
		price_ratios.append(f"\t100-day/250-day:\t{data[12]/data[15]:>9.6f}")
	if data[16] > 0:
		price_ratios.append("x-day/300-day")
		price_ratios.append(f"\t5-day/300-day:\t\t{data[7]/data[16]:>9.6f}")
		price_ratios.append(f"\t10-day/300-day:\t\t{data[8]/data[16]:>9.6f}")
		price_ratios.append(f"\t25-day/300-day:\t\t{data[9]/data[16]:>9.6f}")
		price_ratios.append(f"\t50-day/300-day:\t\t{data[10]/data[16]:>9.6f}")
		price_ratios.append(f"\t75-day/300-day:\t\t{data[11]/data[16]:>9.6f}")
		price_ratios.append(f"\t100-day/300-day:\t{data[12]/data[16]:>9.6f}")
	if data[17] > 0:
		price_ratios.append("x-day/350-day")
		price_ratios.append(f"\t5-day/350-day:\t\t{data[7]/data[17]:>9.6f}")
		price_ratios.append(f"\t10-day/350-day:\t\t{data[8]/data[17]:>9.6f}")
		price_ratios.append(f"\t25-day/350-day:\t\t{data[9]/data[17]:>9.6f}")
		price_ratios.append(f"\t50-day/350-day:\t\t{data[10]/data[17]:>9.6f}")
		price_ratios.append(f"\t75-day/350-day:\t\t{data[11]/data[17]:>9.6f}")
		price_ratios.append(f"\t100-day/350-day:\t{data[12]/data[17]:>9.6f}")
	else:
		price_ratios.append("WARNING: DATA MISSING FROM SMAs; MODEL MAY BE UNRELIABLE")

	price_deltas = ["\nPrice Deltas", 
					"[<0 shows a decrease; >0 shows an increase]", 
					f"5-day -> Present:\t\t{data[0]-data[7]:>9.6f}", 
					f"10-day -> 5-day:\t\t{data[7]-data[8]:>9.6f}", 
					f"25-day -> 10-day:\t\t{data[8]-data[9]:>9.6f}", 
					f"50-day -> 25-day:\t\t{data[9]-data[10]:>9.6f}", 
					f"75-day -> 50-day:\t\t{data[10]-data[11]:>9.6f}", 
					f"100-day -> 75-day:\t\t{data[11]-data[12]:>9.6f}"]
	
	fear_greed_deltas = ["\nFear/Greed Deltas", 
						"[>0 is greedier; <0 is more fearful]", 
						f"3-day -> Present:\t{data[6]-data[18]:>9.6f}", 
						f"5-day -> 3-day:\t\t{data[18]-data[19]:>9.6f}", 
						f"7-day -> 5-day\t\t{data[19]-data[20]:>9.6f}", 
						f"9-day -> 7-day:\t\t{data[20]-data[21]:>9.6f}", 
						f"11-day -> 9-day:\t{data[21]-data[22]:>9.6f}", 
						f"13-day -> 11-day:\t{data[22]-data[23]:>9.6f}", 
						f"15-day -> 13-day:\t{data[23]-data[24]:>9.6f}", 
						f"30-day -> 15-day:\t{data[24]-data[25]:>9.6f}"]

	for item in basic_stats:
		report.append(item)
	for item in price_ratios:
		report.append(item)
	for item in price_deltas:
		report.append(item)
	for item in fear_greed_deltas:
		report.append(item)



def generate_signals():
	report = []

	with open("reports/best_performers.txt") as f:
		best = f.read().splitlines() 

	for coin in dt_agg.coin_id:
		data = pd.read_csv(f"datasets/clean/{coin}_historical_data_clean.csv")
		# extracts the most recent data as a python list
		data = data[data["date"] == str(date.today()-timedelta(0))].values.tolist()[0][1:]
		# stat report
		populate_stat_report(coin, data, report)	

		n_votes = [0, 0, 0, 0, 0] # buy 2x, buy x, hodl, sell y, sell 2y
		n_weights = [0, 0, 0, 0, 0]
		best_model_signal = 6 # set out of bounds to begin with 

		models = {}
		for i in range(len(best)):
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
			# load model
			model = models[filepath]
			model.load_state_dict(torch.load(filepath))
			model.to(torch.device("cpu"))
			# set to prediction mode
			model.eval()
			# make the data pytorch compatible
			feature_tensor = torch.tensor([data], dtype=torch.float32)

			with torch.no_grad():
				output = model(feature_tensor)

			if filepath == best[0]:
				best_model_signal = int(torch.argmax(output, dim=1))
			
			for i in range(len(n_votes)):
				n_weights[i] += float(output[0][i])
				n_votes[int(torch.argmax(output, dim=1))] += 1

		# tabulate answers according to different metrics
		n_votes = torch.tensor(n_votes, dtype=torch.float32)
		n_weights = torch.tensor(n_weights, dtype=torch.float32)
		signal_v = DECISIONS[torch.argmax(n_votes)]
		signal_w = DECISIONS[torch.argmax(n_weights)]
		signal_b = DECISIONS[best_model_signal]
		second_best_w = n_weights[torch.argmin(n_weights)]
		for weight in n_weights:
			if weight > second_best_w and weight < n_weights[torch.argmax(n_weights)]:
				second_best_w = weight

		report.append("\nAction Signals")
		report.append(f"Signal by best nn:\t{signal_b}")
		report.append(f"Signal by votes:\t{signal_v}")
		report.append(f"Signal by weights:\t{signal_w}")
		report.append(f"\tWeights:\t{list(n_weights)}")
		report.append(f"\tDiff 1st and 2nd:\t{(n_weights[torch.argmax(n_weights)] - second_best_w) / len(best):>9.4f}")
		report.append(f"\tDiff 1st and last:\t{(n_weights[torch.argmax(n_weights)] - n_weights[torch.argmin(n_weights)]) / len(best):>9.4f}")


	return report


def generate_report(report):
	with open(f"reports/daily/Daily_Report_{str(date.today())}.txt", "w") as f:
		# starting from index 1 to avoid first triple space divider
		for row in report[1:]:
			f.write(row + "\n")



fetch_data = input("Fetch most recent daily data? [y/n; only if you haven't already fetched today] ")

if (fetch_data.lower())[0] == 'y':
	days_back = -1
	while days_back < 0:
		days_back = int(input("How many days worth of data? [e.g., 5 if you haven't calculated a signal for 5 days] "))
	fetch_new_data(days_back)
	process_new_data()

report = generate_signals()
generate_report(report)

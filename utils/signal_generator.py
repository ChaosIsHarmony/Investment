import torch
import data_aggregator as dt_agg
import data_preprocessor as dt_prepro
import neural_nets as nn
import pandas as pd
import numpy as np
import joblib
import concurrent.futures as cf
from datetime import date, timedelta


DECISIONS = ["BUY", "HODL", "SELL"]
PRICE = 0
MARKET_CAP = 1
VOLUME = 2
FEAR_GREED = 3 
PRICE_5_SMA = 4
PRICE_10_SMA = PRICE_5_SMA+1
PRICE_25_SMA = PRICE_5_SMA+2
PRICE_50_SMA = PRICE_5_SMA+3
PRICE_75_SMA = PRICE_5_SMA+4
PRICE_100_SMA = PRICE_5_SMA+5
PRICE_150_SMA = PRICE_5_SMA+6
PRICE_200_SMA = PRICE_5_SMA+7
PRICE_250_SMA = PRICE_5_SMA+8
PRICE_300_SMA = PRICE_5_SMA+9
PRICE_350_SMA = PRICE_5_SMA+10
FG_3_SMA = 15
FG_5_SMA = FG_3_SMA+1
FG_7_SMA = FG_3_SMA+2
FG_9_SMA = FG_3_SMA+3
FG_11_SMA = FG_3_SMA+4
FG_13_SMA = FG_3_SMA+5
FG_15_SMA = FG_3_SMA+6
FG_30_SMA = FG_3_SMA+7



def fetch_new_data(n_days):
	'''
	Uses multithreading to speed up fetching process.
	'''
	with cf.ThreadPoolExecutor() as executor:
		results = [executor.submit(dt_agg.fetch_missing_data_by_range, coin, n_days) for coin in dt_agg.coin_id]
		
		for thread in cf.as_completed(results):
			print(thread.result())

	with cf.ThreadPoolExecutor() as executor:
		results = [executor.submit(dt_agg.merge_new_dataset_with_old, coin) for coin in dt_agg.coin_id]
		
		for thread in cf.as_completed(results):
			print(thread.result())



def process_individual_coin_new_data(start_date, end_date, coin):
	'''
	Uses multiprocessing to speed up data processing.
	'''
	data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")
	data = dt_prepro.process_data(coin, data, start_date, end_date)
	data.to_csv(f"datasets/clean/{coin}_historical_data_clean.csv", index=False)

	return f"All new data processed for {coin}."


def process_new_data():
	start_date = str(date.today())
	end_date = "2020-08-23" #the first day of data of: the youngest asset: polkadot
	with cf.ProcessPoolExecutor() as executor:
		results = [executor.submit(process_individual_coin_new_data, start_date, end_date, coin) for coin in dt_agg.coin_id]

		for process in cf.as_completed(results):
			print(process.result())
	

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



def populate_stat_report_full(coin, data, raw_data, report):
	basic_stats = ["\n\n\n________________________________________", 
					f"Report for {coin.upper()}:", 
					"Basic Stats", 
					"[1.0 is the highest; 0.0 is the lowest]", 
					f"price:\t\t{data[PRICE]:.6f}", 
					f"market_cap:\t{data[MARKET_CAP]:.6f}", 
					f"volume:\t\t{data[VOLUME]:.6f}", 
					f"fear/greed\t{data[FEAR_GREED]:.6f} [{get_fg_indicator(data[FEAR_GREED])}]"]
	
	price_ratios = ["\nPrice Ratios", 
					"[>0 means greater risk/overvalued; <0 means less risk/undervalued]", 
					f"5-day/10-day:\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_10_SMA]:>9.6f}", 
					f"10-day/25-day:\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_25_SMA]:>9.6f}", 
					f"25-day/50-day:\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_50_SMA]:>9.6f}"]

	if raw_data[PRICE_200_SMA] > 0:
		price_ratios.append("x-day/200-day")
		price_ratios.append(f"\t5-day/200-day:\t\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
		price_ratios.append(f"\t10-day/200-day:\t\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
		price_ratios.append(f"\t25-day/200-day:\t\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
		price_ratios.append(f"\t50-day/200-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
		price_ratios.append(f"\t75-day/200-day:\t\t{raw_data[PRICE_75_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
		price_ratios.append(f"\t100-day/200-day:\t{raw_data[PRICE_100_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
		price_ratios.append(f"\t150-day/200-day:\t{raw_data[PRICE_150_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
	if raw_data[PRICE_250_SMA] > 0:
		price_ratios.append("x-day/250-day")
		price_ratios.append(f"\t5-day/250-day:\t\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
		price_ratios.append(f"\t10-day/250-day:\t\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
		price_ratios.append(f"\t25-day/250-day:\t\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
		price_ratios.append(f"\t50-day/250-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
		price_ratios.append(f"\t75-day/250-day:\t\t{raw_data[PRICE_75_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
		price_ratios.append(f"\t100-day/250-day:\t{raw_data[PRICE_100_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
		price_ratios.append(f"\t150-day/250-day:\t{raw_data[PRICE_150_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
	if raw_data[PRICE_300_SMA] > 0:
		price_ratios.append("x-day/300-day")
		price_ratios.append(f"\t5-day/300-day:\t\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
		price_ratios.append(f"\t10-day/300-day:\t\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
		price_ratios.append(f"\t25-day/300-day:\t\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
		price_ratios.append(f"\t50-day/300-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
		price_ratios.append(f"\t75-day/300-day:\t\t{raw_data[PRICE_75_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
		price_ratios.append(f"\t100-day/300-day:\t{raw_data[PRICE_100_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
		price_ratios.append(f"\t150-day/300-day:\t{raw_data[PRICE_150_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
	if raw_data[PRICE_350_SMA] > 0:
		price_ratios.append("x-day/350-day")
		price_ratios.append(f"\t5-day/350-day:\t\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
		price_ratios.append(f"\t10-day/350-day:\t\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
		price_ratios.append(f"\t25-day/350-day:\t\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
		price_ratios.append(f"\t50-day/350-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
		price_ratios.append(f"\t75-day/350-day:\t\t{raw_data[PRICE_75_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
		price_ratios.append(f"\t100-day/350-day:\t{raw_data[PRICE_100_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
		price_ratios.append(f"\t150-day/350-day:\t{raw_data[PRICE_150_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
	else:
		price_ratios.append("WARNING: DATA MISSING FROM SMAs; MODEL MAY BE UNRELIABLE")

	price_deltas = ["\nPrice Deltas", 
					"[<0 shows a decrease; >0 shows an increase]", 
					f"5-day -> Present:\t\t{data[PRICE]-data[PRICE_5_SMA]:>9.6f}", 
					f"10-day -> 5-day:\t\t{data[PRICE_5_SMA]-data[PRICE_10_SMA]:>9.6f}", 
					f"25-day -> 10-day:\t\t{data[PRICE_10_SMA]-data[PRICE_25_SMA]:>9.6f}", 
					f"50-day -> 25-day:\t\t{data[PRICE_25_SMA]-data[PRICE_50_SMA]:>9.6f}", 
					f"75-day -> 50-day:\t\t{data[PRICE_50_SMA]-data[PRICE_75_SMA]:>9.6f}", 
					f"100-day -> 75-day:\t\t{data[PRICE_75_SMA]-data[PRICE_100_SMA]:>9.6f}"]
	
	fear_greed_deltas = ["\nFear/Greed Deltas", 
						"[>0 is greedier; <0 is more fearful]", 
						f"3-day -> Present:\t{data[FEAR_GREED]-data[FG_3_SMA]:>9.6f}", 
						f"5-day -> 3-day:\t\t{data[FG_3_SMA]-data[FG_5_SMA]:>9.6f}", 
						f"7-day -> 5-day\t\t{data[FG_5_SMA]-data[FG_7_SMA]:>9.6f}", 
						f"9-day -> 7-day:\t\t{data[FG_7_SMA]-data[FG_9_SMA]:>9.6f}", 
						f"11-day -> 9-day:\t{data[FG_9_SMA]-data[FG_11_SMA]:>9.6f}", 
						f"13-day -> 11-day:\t{data[FG_11_SMA]-data[FG_13_SMA]:>9.6f}", 
						f"15-day -> 13-day:\t{data[FG_13_SMA]-data[FG_15_SMA]:>9.6f}", 
						f"30-day -> 15-day:\t{data[FG_15_SMA]-data[FG_30_SMA]:>9.6f}"]

	for item in basic_stats:
		report.append(item)
	for item in price_ratios:
		report.append(item)
	for item in price_deltas:
		report.append(item)
	for item in fear_greed_deltas:
		report.append(item)



def populate_stat_report_essentials(coin, data, raw_data, report):
	basic_stats = ["\n\n\n________________________________________", 
					f"Report for {coin.upper()}:", 
					"Basic Stats", 
					"[1.0 is the highest; 0.0 is the lowest]", 
					f"price:\t\t{data[PRICE]:.6f}", 
					f"market_cap:\t{data[MARKET_CAP]:.6f}", 
					f"volume:\t\t{data[VOLUME]:.6f}", 
					f"fear/greed\t{data[FEAR_GREED]:.6f} [{get_fg_indicator(data[FEAR_GREED])}]"]
	
	price_ratios = ["\nPrice Ratios", 
					"[>0 means greater risk/overvalued; <0 means less risk/undervalued]"] 

	if raw_data[PRICE_150_SMA] > 0:
		price_ratios.append(f"\t50-day/150-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_150_SMA]:>9.6f}")
	if raw_data[PRICE_200_SMA] > 0:
		price_ratios.append(f"\t50-day/200-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
	if raw_data[PRICE_250_SMA] > 0:
		price_ratios.append(f"\t50-day/250-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
	if raw_data[PRICE_300_SMA] > 0:
		price_ratios.append(f"\t50-day/300-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
	if raw_data[PRICE_350_SMA] > 0:
		price_ratios.append(f"\t50-day/350-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
	else:
		price_ratios.append("WARNING: DATA MISSING FROM SMAs; MODEL MAY BE UNRELIABLE")

	price_deltas = ["\nPrice Deltas", 
					"[<0 shows a decrease; >0 shows an increase]", 
					f"25-day -> Present:\t\t{data[PRICE]-data[PRICE_25_SMA]:>9.6f}", 
					f"50-day -> Present:\t\t{data[PRICE]-data[PRICE_50_SMA]:>9.6f}", 
					f"100-day -> Present:\t\t{data[PRICE]-data[PRICE_100_SMA]:>9.6f}"]
	
	fear_greed_deltas = ["\nFear/Greed Deltas", 
						"[>0 is greedier; <0 is more fearful]", 
						f"7-day -> Present:\t{data[FEAR_GREED]-data[FG_7_SMA]:>9.6f}", 
						f"15-day -> Present:\t{data[FEAR_GREED]-data[FG_15_SMA]:>9.6f}", 
						f"30-day -> Present:\t{data[FEAR_GREED]-data[FG_30_SMA]:>9.6f}"]

	for item in basic_stats:
		report.append(item)
	for item in price_ratios:
		report.append(item)
	for item in price_deltas:
		report.append(item)
	for item in fear_greed_deltas:
		report.append(item)


def load_model(neural_net, filepath):
	model = neural_net
	model.load_state_dict(torch.load(filepath))


	return model



def get_models(best):
	models = []
	nn.set_model_parameters()
	for i in range(len(best)):
		if "Laptop_0" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Laptop_0(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Laptop_1" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Laptop_1(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Laptop_2" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Laptop_2(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Laptop_3" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Laptop_3(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Laptop_4" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Laptop_4(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Laptop_5" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Laptop_5(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Laptop_6" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Laptop_6(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Pi_0" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Pi_0(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Pi_1" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Pi_1(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Pi_2" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Pi_2(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Pi_3" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Pi_3(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Pi_4" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Pi_4(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Pi_5" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Pi_5(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Pi_6" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Pi_6(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "Pi_7" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_Pi_7(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "PC_0" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_PC_0(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "PC_1" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_PC_1(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "PC_2" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_PC_2(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "PC_3" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_PC_3(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "PC_4" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_PC_4(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "PC_5" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_PC_5(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "PC_6" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_PC_6(nn.N_FEATURES, nn.N_SIGNALS), best[i]))
		elif "PC_7" in best[i]:
			models.append(load_model(nn.CryptoSoothsayer_PC_7(nn.N_FEATURES, nn.N_SIGNALS), best[i]))


	return models



def get_signal_strength(data, raw_data, buy_signal, hodl_signal, sell_signal):
	# direction of signal
	buy = buy_signal > sell_signal

	# ratio of buy to sell or sell to buy depending on signal direction
	signal_strength = (abs(buy_signal - hodl_signal) / abs(sell_signal - hodl_signal)) if buy else (abs(sell_signal - hodl_signal) / abs(buy_signal - hodl_signal))
	
	# counter to typical action (Buy when fearful; Sell when greedy), thus detracts; else, additive
	fg_mult = -1 if (buy and data[FEAR_GREED] >= 0.5) or (not buy and data[FEAR_GREED] <= 0.5) else 1

	signal_strength += fg_mult * ((0.5 - data[FEAR_GREED]) / 0.5) if buy else fg_mult * ((FEAR_GREED - 0.5) / 0.5)
	
	# under/overvaluation
	tot_val = 0
	tot_metrics = 0
	if raw_data[PRICE_200_SMA] > 0:
		tot_val += 1 / (raw_data[PRICE_50_SMA] / raw_data[PRICE_200_SMA]) if buy else (raw_data[PRICE_50_SMA] / raw_data[PRICE_200_SMA])
		tot_metrics += 1

	if raw_data[PRICE_250_SMA] > 0:
		tot_val += 1 / (raw_data[PRICE_50_SMA] / raw_data[PRICE_250_SMA]) if buy else (raw_data[PRICE_50_SMA] / raw_data[PRICE_250_SMA])
		tot_metrics += 1

	if raw_data[PRICE_300_SMA] > 0:
		tot_val += 1 / (raw_data[PRICE_50_SMA] / raw_data[PRICE_300_SMA]) if buy else (raw_data[PRICE_50_SMA] / raw_data[PRICE_300_SMA])
		tot_metrics += 1

	if raw_data[PRICE_350_SMA] > 0:
		tot_val += 1 / (raw_data[PRICE_50_SMA] / raw_data[PRICE_350_SMA]) if buy else (raw_data[PRICE_50_SMA] / raw_data[PRICE_350_SMA])
		tot_metrics += 1

	signal_strength += 0 if tot_metrics == 0 else fg_mult * (tot_val / tot_metrics)

	# ultimate decision
	buy_or_sell = "BUY" if buy else "SELL"

	return signal_strength, buy_or_sell



def generate_signals(full_report=False):
	report = []

	with open("reports/best_performers.txt") as f:
		best_models = f.read().splitlines() 

	for coin in dt_agg.coin_id:
		# NOTE: raw_data is used for the SMA ratio calculations as the normalized data cannot adequately capture the ratios' significances
		data = pd.read_csv(f"datasets/clean/{coin}_historical_data_clean.csv")
		raw_data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw_all_features.csv")
		# extracts the most recent data as a python list
		data = data[data["date"] == str(date.today()-timedelta(0))].values.tolist()[0][1:-1]
		raw_data = raw_data[raw_data["date"] == str(date.today()-timedelta(0))].values.tolist()[0][1:-1]
		# stat report
		if full_report:
			populate_stat_report_full(coin, data, raw_data, report)
		else:
			populate_stat_report_essentials(coin, data, raw_data, report)	

		n_votes = [0, 0, 0] # buy 3x, buy 2x, buy x, hodl, sell y, sell 2y, sell 3y
		n_weights = [0, 0, 0]
		best_model_signal = 3 # set out of bounds to begin with 

		# get the best performing models
		models = get_models(best_models)

		for i in range(len(models)):
			# set to prediction mode
			model = models[i]
			model.eval()
			# make the data pytorch compatible
			feature_tensor = torch.tensor([data], dtype=torch.float32)

			with torch.no_grad():
				output = model(feature_tensor)

			if i == 0:
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

		formatted_w_list = [round((x/len(best_models)), 4) for x in n_weights.tolist()]	
		
		best_w = max(formatted_w_list)
		second_best_w = min(formatted_w_list) # placeholder
		worst_w = min(formatted_w_list)
		
		buy_signal = formatted_w_list[0]
		hodl_signal = formatted_w_list[1]
		sell_signal = formatted_w_list[2]
		
		# calculate signal strength
		signal_strength, buy_or_sell = get_signal_strength(data, raw_data, buy_signal, hodl_signal, sell_signal)

		for weight in formatted_w_list:
			if weight > second_best_w and weight < best_w:
				second_best_w = weight

		report.append("\nAction Signals")
		report.append(f"Signal by best nn:\t{signal_b}")
		report.append(f"Signal by votes:\t{signal_v}")
		report.append(f"Signal by weights:\t{signal_w}")
		report.append(f"Buy/Sell pressure:\t{buy_or_sell} {signal_strength:.1f}X")
		report.append("\t[Greater disparities mean a more confident signal]")
		report.append(f"\tWeights:\t{formatted_w_list}")
		report.append(f"\tDiff 1st and 2nd:\t{best_w - second_best_w:>9.4f}")
		report.append(f"\tDiff 1st and last:\t{best_w - worst_w:>9.4f}")
		report.append(f"\tDiff BUY and SELL:\t{abs(buy_signal - sell_signal):>9.4f}")


	return report


def generate_report(report):
	with open(f"reports/daily/Daily_Report_{str(date.today())}.txt", "w") as f:
		# starting from index 1 to avoid first triple space divider
		for row in report[1:]:
			f.write(row + "\n")



def main():
	# collects new data and then cleans it
	fetch_data = input("Fetch most recent daily data? [y/n; only if you haven't already fetched today] ")
	if (fetch_data.lower())[0] == 'y':
		days_back = -1
		while days_back < 0:
			days_back = int(input("How many days worth of data? [e.g., 5 if you haven't calculated a signal for 5 days] "))
		fetch_new_data(days_back)
		process_new_data()

	# determines signal and creates report
	full_report = input("Full report? [y/n; y gives all the gory details] ")
	if (full_report.lower())[0] == 'y':
		report = generate_signals(True)
	else:
		report = generate_signals()

	generate_report(report)



if __name__ == "__main__":
	main()

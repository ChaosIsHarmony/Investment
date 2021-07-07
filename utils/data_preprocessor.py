import pandas as pd


def handle_missing_data(data, start_date, end_date):
	'''
	Checks for missing days
	Fills all NaN values with 0.
	Takes average of prior day and next day to calculate missing value or previous or next day if data point is at the beginning or end of the dataset, respectively.
	'''	
	# check for missing dates
	data["date"] = pd.to_datetime(data["date"])
	missing_dates = pd.date_range(start = start_date, end = end_date ).difference(data["date"])
	assert len(missing_dates) == 0, f"The dataset is missing dates: {missing_dates}. Use utils/data_aggregator.py to collect the missing dates before proceeding."

	data = data.fillna(0)

	numeric_data = data.iloc[:, 1:]

	for i, row in data.iterrows():
		for column in data.columns[1:]:
			if row[column] == 0:
				next_non_zero = 0
				start_ind = i + 1
				while next_non_zero == 0 and start_ind < data.shape[0]:
					next_non_zero += data.loc[start_ind, column]
					start_ind += 1
				
				# leading zero
				if i == 0:
					data.loc[i, column] = next_non_zero
				# Take average of two closest data points
				elif next_non_zero > 0:
					data.loc[i, column] = (data.loc[i-1, column] + next_non_zero) / 2
				# Otherwise, just make same as one before it
				elif i > 0:
					data.loc[i, column] = data.loc[i-1, column]

	return data



def normalize_data(data):
	'''
	Normalizes data using min-max normalization but only up until the given point in history, e.g., datapoint for 2021/02/28 does not have any knowledge of data from 01/03/2021 and onwards.
	NOTE: only uses 2 until seocnd to last column, because 1st column is the date and the last column is the signal (i.e., what the appropriate action to take should be).
	'''
	data_cp = data.copy(deep=True)

	# 1st row items only have themselves as history
	for column in range(1, data.shape[1]-1):
		data_cp.iloc[0, column] = 1.0

	for i in range(1, data.shape[0]):
		for column in range(1, data.shape[1]-1):
			col_max = data.iloc[:i+1, column].max()
			col_min = data.iloc[:i+1, column].min()
			# to avoid division by zero
			if col_max == col_min and col_max != 0:
				col_min -= 1
			elif col_max == 0:
				col_max += 1
			data_cp.iloc[i, column] = (data.iloc[i, column] - col_min) / (col_max - col_min)

	# last row
	for column in range(1, data.shape[1]-1):
		col_max = data.iloc[:, column].max()
		col_min = data.iloc[:, column].min()
		if col_max == col_min and col_max != 0:
			col_min -= 1
		elif col_max == 0:
			col_max += 1
		data_cp.iloc[-1, column] = (data.iloc[-1, column] - col_min) / (col_max - col_min)

	# fear and greed index is out of 100
	for column in data.columns:
		if "fear_greed" in str(column):
			data_cp[column] = data[column] / 100

	# in case there were division by zero errors leading to NaN
	data_cp = data_cp.fillna(0)

	return data_cp



def perform_SMA_calculation(data, totals, SMAs, column_name):
	for i, row in data.iterrows():
		for key in totals.keys():
			totals[key] += data.loc[i, column_name]
			# can't do i+1 because (i+1)-key would leave only 4 days in the total for the 5-day MA, 9 for the 10-day, etc. and would make the calculation inaccurate
			if key <= i:
				totals[key] -= data.loc[i-key, column_name]
				SMAs[key][i] = totals[key] / key
	
	for key in SMAs.keys():
		data[f"{column_name}_{key}_SMA"] = SMAs[key]

	return data



def generate_price_SMA_lists_dict(list_size):
	SMAs = {5: [], 10:[], 25: [], 50: [], 75: [], 100: [], 150: [], 200: [], 250: [], 300: [], 350: []}
	
	for i in range(list_size):
		for key in SMAs.keys():
			SMAs[key].append(0)

	return SMAs



def generate_fear_greed_SMA_lists_dict(list_size):
	SMAs = {3: [], 5:[], 7: [], 9: [], 11: [], 13: [], 15: [], 30: []}
	
	for i in range(list_size):
		for key in SMAs.keys():
			SMAs[key].append(0)

	return SMAs



def calculate_price_SMAs(data):
	'''
	Calculates Simple Moving Averages to the maximum extent allowed by the data
	'''
	n_datapoints = data.shape[0]
	totals = {5:0, 10:0, 25:0, 50:0, 75:0, 100:0, 150:0, 200:0, 250:0, 300:0, 350:0}
	SMAs = generate_price_SMA_lists_dict(n_datapoints)

	data = perform_SMA_calculation(data, totals, SMAs, "price")
	
	return data



def calculate_fear_greed_SMAs(data):
	'''
	Calculates Simple Moving Averages for Fear/Greed index over several discrete intervals for the past fortnight
	'''
	n_datapoints = data.shape[0]
	totals = {3:0, 5:0, 7:0, 9:0, 11:0, 13:0, 15:0, 30:0}
	SMAs = generate_fear_greed_SMA_lists_dict(n_datapoints)
	
	data = perform_SMA_calculation(data, totals, SMAs, "fear_greed")

	return data



def calculate_SMAs(data):
	# reverse the dataframe for easier calculation logic
	data = data.reindex(index=data.index[::-1]).reset_index()
	data = data.drop(columns=["index"])

	# price
	data = calculate_price_SMAs(data)
	# fear/greed index
	data = calculate_fear_greed_SMAs(data)

	return data



def get_signal_value(percent_delta):
	'''
	Returns signal to BUY, SELL, or HODL [0-3 scale] based on percent_delta over given period (as determined by the calling method).
	'''
	signal = 1 # HODL by default

	# SELL - lower is stronger
	if percent_delta < -0.1:
		signal = 2 
	# BUY - higher is stronger
	elif percent_delta > 0.1:
		signal = 0

	return signal



def get_weighting_constant(n=28):
	'''
	Calculates weighting constant required for the given period so that all weights add to 1.
	NOTE: simplified from (100 / ((n * (n+1)) / 2)) / 100, where the numerator provides a constant that when multiplied by unit increments, the addition of all terms from 1 -> n will add to 1.
	'''
	return 2 / (n*(n+1))



def calculate_signals(data, limit=28):
	'''
	Calculates the signal on a scale from 0-3 (BUY, HODL, & SELL) based on weighted average of the price future (days_out) price movement deltas.
	If percentage increase exceeds given threshold, then SELL; if decrease then BUY.
	If percentage increase/decrease does not exceed minimum thresholds, then HODL.
	NOTE: Weights days in the more distant future more heavily.
	'''
	signals = []
	weighting_constant = get_weighting_constant(limit)
	for ind in range(len(data) - limit):
		current_price = data["price"][ind]
		price_delta_avg = 0.0
		for days_from_now in range(1, limit+1):
			later_price = data["price"][ind+days_from_now]
			percent_delta = (later_price - current_price) / current_price
			price_delta_avg += percent_delta * weighting_constant * days_from_now
		signals.append(get_signal_value(price_delta_avg))

	data["signal"] = pd.Series(signals)

	print("Value counts for signals in dataset:\n", data["signal"].value_counts())

	data = data.fillna(0)

	return data



def process_data(data, start_date, end_date):
	'''
	Processes the basic data provided by coingecko in the following ways:
		
		- Fills in missing values
		- Calculates Simple Moving Averages for a variety of intervals
		- Calculates the signal for that day
			- Prescient looking forward x-days and averaging the price_deltas
		- Normalizes all values by dividing by the max value in each category
			- Normalizes neither date nor signal columns
	'''
	# Fill in missing values
	data = handle_missing_data(data, start_date, end_date)
	print("Missing data handling complete.")
	# Calculate SMAs 
	data = calculate_SMAs(data)
	print("SMA calculation complete.")
	# Calculate signals
	days_from_now = 7 * 5 # 7 * n weeks 
	data = calculate_signals(data, days_from_now)
	print(f"Signal calculation for {days_from_now} days from now complete.")
	# Normalize, must happen after SMA calculation or will skew results
	data = normalize_data(data)
	print("Data normalization complete.")
	print()

	return data



def run():
	coins = ["aave", "algorand", "bitcoin", "cardano", "chainlink", "cosmos", "decentraland", "ethereum", "matic-network", "the-graph", "theta-token"]  
	# The following two coins have shorter histories and require a different start date {polkadot = 2020-08-23; solana = 2020-04-11}
	#coins = ["polkadot"]
	#coins = ["solana"]

	start_date = "2020-12-25"#"2019-10-20"
	end_date = "2021-06-11"

	for coin in coins:
		print(coin)

		data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")

		data = process_data(data, start_date, end_date)

		data.to_csv(f"datasets/clean/{coin}_historical_data_clean.csv", index=False)



#run()

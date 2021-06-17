import pandas as pd


def handle_missing_data(data, start_date, end_date):
	'''
	Checks for missing days
	Fills all NaN values with 0.
	Takes average of prior day and next day to calculate missing value or previous or next day if data point is at the beginning or end of the dataset, respectively.
	'''	
	data["date"] = pd.to_datetime(data["date"])
	missing_dates = pd.date_range(start = start_date, end = end_date ).difference(data["date"])
	assert len(missing_dates) == 0, f"The dataset is missing dates: {missing_dates}. Use utils/data_aggregator.py to collect the missing dates before proceeding."

	data = data.fillna(0)

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
	Normalizes data using min-max normalization, as Z-score normalization would be more suited to data with outliers.
	Starts from second column to avoid normalizes the date column.
	'''
	for column in data.columns[1:]:
		data[column] = (data[column]-data[column].min()) / (data[column].max() - data[column].min())

	return data

def generate_price_SMA_lists_dict(list_size):
	SMAs = {5: [], 10:[], 25: [], 50: [], 75: [], 100: [], 150: [], 200: [], 250: [], 300: [], 350: []}
	
	for i in range(list_size):
		for key in SMAs.keys():
			SMAs[key].append(0)

	return SMAs


def calculate_price_SMAs(data):
	'''
	Calculates Simple Moving Averages to the maximum extent allowed by the data
	'''
	# reverse the dataframe for easier calculation logic
	data = data.reindex(index=data.index[::-1]).reset_index()
	data = data.drop(columns=["index"])

	n_datapoints = data.shape[0]
	totals = {5:0, 10:0, 25:0, 50:0, 75:0, 100:0, 150:0, 200:0, 250:0, 300:0, 350:0}
	SMAs = generate_price_SMA_lists_dict(n_datapoints)
	
	for i, row in data.iterrows():
		for key in totals.keys():
			totals[key] += data.loc[i, "price"]
			# can't do i+1 because (i+1)-key would leave only 4 days in the total for the 5-day MA, 9 for the 10-day, etc. and would make the calculation inaccurate
			if key <= i:
				totals[key] -= data.loc[i-key, "price"]
				SMAs[key][i] = totals[key] / key
	
	for key in SMAs.keys():
		data[f"price_{key}_SMA"] = SMAs[key]

	return data


def generate_fear_greed_SMA_lists_dict(list_size):
	SMAs = {3: [], 5:[], 7: [], 9: [], 11: [], 13: [], 15: []}
	
	for i in range(list_size):
		for key in SMAs.keys():
			SMAs[key].append(0)

	return SMAs


def calculate_fear_greed_SMAs(data):
	'''
	Calculates Simple Moving Averages for Fear/Greed index over several discrete intervals for the past fortnight
	'''
	# reverse the dataframe for easier calculation logic
	data = data.reindex(index=data.index[::-1]).reset_index()
	data = data.drop(columns=["Unnamed: 0", "index"])

	n_datapoints = data.shape[0]
	totals = {3:0, 5:0, 7:0, 9:0, 11:0, 13:0, 15:0}
	SMAs = generate_fear_greed_SMA_lists_dict(n_datapoints)
	
	for i, row in data.iterrows():
		for key in totals.keys():
			totals[key] += data.loc[i, "fear_greed"]
			if key <= i:
				totals[key] -= data.loc[i-key, "fear_greed"]
				SMAs[key][i] = totals[key] / key
	
	for key in SMAs.keys():
		data[f"fear_greed_{key}_SMA"] = SMAs[key]

	return data


def calculate_SMAs(data):
	# price
	data = calculate_price_SMAs(data)
	# fear/greed index
	data = calculate_fear_greed_SMAs(data)

	return data


def process_data(data, start_date, end_date):
	'''
	Processes the basic data provided by coingecko in the following ways:
		
		- Fills in missing values
		- Normalizes all values by dividing by the max value in each category
		- Calculates Simple Moving Averages for a variety of intervals
	'''
	# Fill in missing values
	data = handle_missing_data(data, start_date, end_date)
	print("Missing data handling complete.")
	# Normalize
	data = normalize_data(data)
	print("Data normalization complete.")
	# Calculate SMAs 
	data = calculate_SMAs(data)
	print("SMA calculation complete.")

	return data


def run():
#	coins = ["algorand", "bitcoin", "cardano", "chainlink", "cosmos", "ethereum", "matic-network", "polkadot", "solana", "theta-token"]

	coins = ["bitcoin"]
	start_date = "2019-10-20"
	end_date = "2021-06-11"

	for coin in coins:
		print(coin)

		data = pd.read_csv(f"datasets/raw/{coin}_historical_data.csv")

		data = process_data(data, start_date, end_date)

		data.to_csv(f"datasets/clean/{coin}_historical_data_clean.csv", index=False)



#run()

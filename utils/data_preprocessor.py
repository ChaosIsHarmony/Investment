import pandas as pd


def handle_missing_data(data):
	'''
	Fills all NaN values with 0
	Takes average of prior day and next day to calculate missing value
	'''	
	data = data.fillna(0)

	for i, row in data.iterrows():
		for column in data.columns[1:]:
			if row[column] == 0:
				next_non_zero = 0
				start_ind = i + 1
				while next_non_zero == 0 and start_ind < data.shape[0]:
					next_non_zero += data.loc[start_ind, column]
					start_ind += 1
					
				# Take average of two closest data points
				if next_non_zero > 0:
					data.loc[i, column] = (data.loc[i-1, column] + next_non_zero) / 2
				# Otherwise, just make same as one before it
				elif i > 0:
					data.loc[i, column] = data.loc[i-1, column]

	return data


def normalize_data(data):
	'''
	Normalizes data using min-max normalization, as Z-score normalization would be more suited to data with outliers
	'''
	for column in data.columns[2:]:
		data[column] = (data[column]-data[column].min()) / (data[column].max() - data[column].min())

	return data

def generate_SMA_lists_dict(list_size):
	SMAs = {5: [], 10:[], 25: [], 50: [], 75: [], 100: [], 150: [], 200: [], 250: [], 300: [], 350: []}
	
	for i in range(list_size):
		for key in SMAs.keys():
			SMAs[key].append(0)

	return SMAs


def calculate_SMAs(data):
	'''
	Calculates Simple Moving Averages to the maximum extent allowed by the data
	'''
	# reverse the dataframe for easier calculation logic
	data = data.reindex(index=data.index[::-1]).reset_index()
	data = data.drop(columns=["Unnamed: 0", "index"])

	n_datapoints = data.shape[0]
	totals = {5:0, 10:0, 25:0, 50:0, 75:0, 100:0, 150:0, 200:0, 250:0, 300:0, 350:0}
	SMAs = generate_SMA_lists_dict(n_datapoints)
	
	for i, row in data.iterrows():
		for key in totals.keys():
			totals[key] += data.loc[i, "price"]
			if key <= i:
				totals[key] -= data.loc[i-key, "price"]
				SMAs[key][i] = totals[key] / key
	
	for key in SMAs.keys():
		data[f"{key}_SMA"] = SMAs[key]

	return data


def process_data(data):
	'''
	Processes the basic data provided by coingecko in the following ways:
		
		- Fills in missing values
		- Normalizes all values by dividing by the max value in each category
		- Calculates Simple Moving Averages for a variety of intervals
	'''
	# Fill in missing values
	data = handle_missing_data(data)
	print("Missing data handling complete.")
	# Normalize
	data = normalize_data(data)
	print("Data normalization complete.")
	# Calculate SMAs 
	data = calculate_SMAs(data)
	print("SMA calculation complete.")

	return data


def run():
	coins = ["algorand", "bitcoin", "cardano", "chainlink", "cosmos", "ethereum", "matic-network", "polkadot", "solana", "theta-token"]

	for coin in coins:
		print(coin)

		data = pd.read_csv(f"datasets/raw/{coin}_historical_data.csv")

		data = process_data(data)

		data.to_csv(f"datasets/clean/{coin}_historical_data_clean.csv")


should_run = input("Run the data_preprocessor and create new datasets? ")

if should_run[0] == 'y' or should_run[0] == 'Y':
	run()

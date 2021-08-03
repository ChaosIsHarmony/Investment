import pandas as pd
import os
from typing import List


def merge_datasets(coin: str, list_of_datasets: List[pd.DataFrame], all_data: bool = False) -> None:
	'''
	Merges two or more datasets.
	Param all_data is used when combining all datasets into one mega dataset.
	
	NOTE: Param list_of_datasets must be a list of pandas DataFrames
	'''
	merged_data = pd.concat(list_of_datasets)
	merged_data["date"] = pd.to_datetime(merged_data["date"], dayfirst=True, infer_datetime_format=True)
	merged_data = merged_data.sort_values(by=["date"], ascending=False)
	# if merging all datasets into mega training dataset, more than date must be unique
	if all_data:
		subset_cols = ["date", "price", "market_cap", "volume"]
	else:
		subset_cols = ["date"]

	merged_data = merged_data.drop_duplicates(subset=subset_cols, keep="last")
	merged_data = merged_data.reset_index()
	merged_data = merged_data.drop(columns=["index"])

	if all_data:
		merged_data.to_csv(f"datasets/complete/all_historical_data_complete.csv", index=False, float_format="%f")
	else:
		merged_data.to_csv(f"datasets/raw/{coin}_historical_data_raw.csv", index=False, float_format="%f")



def merge_new_dataset_with_old(coin: str, by_range: bool = True) -> str:
	'''
	Merges all previous datasets with the newly fetched data.
	NOTE: Assumes fetch_missing_data_by_range or fetch_missing_data_by_date have been called first.
	'''
	data_to_merge = [pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")]

	if by_range:
		data_to_merge.append(pd.read_csv(f"datasets/raw/{coin}_historical_data_by_range.csv"))
		os.remove(f"datasets/raw/{coin}_historical_data_by_range.csv")
	else:
		data_to_merge.append(pd.read_csv(f"datasets/raw/{coin}_historical_data_by_date.csv"))
		os.remove(f"datasets/raw/{coin}_historical_data_by_date.csv")

	merge_datasets(coin, data_to_merge)

	return f"Successfully merged new and old {coin} data"



def combine_all_datasets(coins):
	'''
	Combines all datasets that have already been prepared for training.
	'''
	datasets_to_merge = []
	for coin in coins:
		# Currently polkadot has fewer than 350 days worth of datasets
		# In the future, I can delete this line
		if coin != "polkadot":
			datasets_to_merge.append(pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv"))

	merge_datasets("all", datasets_to_merge, all_data=True)





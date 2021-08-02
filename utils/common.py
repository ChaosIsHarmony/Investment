'''
Houses all the variables and methods shared among files (DRY).
Houses interface for all sublibraries so that when modifying sublibrary, will not have to modify its method names in all other sub libraries.
'''
import data_aggregator as dt_agg
import data_preprocessor as dt_pp
import data_processor as dt_p
import dataset_combiner as dt_c
import risk_adjusted_return_calculator as rarc
import neural_nets as nn
import param_trainer_parser as ptp

#
# ------------- CONSTANTS ------------
#
SIGNAL_FOR_N_DAYS_FROM_NOW = 7 * 5 # 7 * n weeks 
PROMISING_ACCURACY_THRESHOLD = 0.655
OUTSTANDING_ACCURACY_THRESHOLD = 0.76
INACCURACY_THRESHOLD = 0.05

coins = ["algorand", "bitcoin", "cardano", "chainlink", "ethereum", "polkadot", "solana"]
possible_coins = ["harmony", "matic-network", "quant-network", "theta-token", "zilliqa"]

#
# ------------- INTERFACES ------------
#
# DATA AGGREGATOR
def aggregate_data_for_new_coins(coins, interval=600):
	dt_agg.aggregate_data_for_new_coins(coins, interval)


def aggregate_new_data(coin, n_days):
	return dt_agg.fetch_missing_data_by_range(coin, n_days)



# DATA PREPROCESSOR
def clean_coin_data(coin, data, start_date, end_date):
	return dt_pp.process_data(coin, data, start_date, end_date)


def handle_missing_data(data, start_date, end_date):
    return dt_pp.handle_missing_data(data, start_date, end_date)



# DATASET COMBINER
def merge_newly_aggregated_data(coin, by_range=True):
	return dt_c.merge_new_dataset_with_old(coin, by_range)


def merge_datasets(coin, datasets, all_data):
	dt_c.merge_datasets(coin, datasets, all_data)



# NEURAL NETS
def load_nn_model(filepath):
	return nn.load_model(filepath)


def set_nn_model_parameters():
	nn.set_model_parameters()



# PARAMATER TRAINER PARSER
def get_model_params(coin, filepath):
	return ptp.get_model_params(coin, filepath)


def parse_training_reports(coin, model_architecture):
	return ptp.parse_reports(coin, model_architecture)



# RISK ADJUSTED RETURN CALCULATOR
def get_sharpe_ratio(coin):
	return rarc.get_sharpe_ratio(coin)


def get_upi(coin):
	return rarc.get_upi(coin)

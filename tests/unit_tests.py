'''
RUN WITH: $ python3 -m unittest tests.unit_tests
'''
import src 
from src import investstrat as strat
import utils
from utils import data_preprocessor as dpp
from utils import data_processor as dp
from utils import neural_nets as nn

import copy
import pandas as pd

def run_tests():
	prices = []
	for i in range(500):
		prices.append([i,float(i+1)])
	test_find_max(prices)
	print("test_find_max() tests all passed")
	test_normalize(prices)
	print("test_normalize() tests all passed")
	test_calc_SMA(prices)
	print("test_calc_SMA() tests all passed")

	# DATA PRE-PROCESSOR TESTS
	test_handle_missing_data()
	print("test_handle_missing_data() tests all passed")
	test_normalize_data()
	print("test_normalize_data() tests all passed")
	test_calculate_SMAs()
	print("test_calculate_SMAs() tests all passed")

	# DATA PROCESSOR TESTS
	test_generate_dataset()
	print("test_generate_dataset() tests all passed")


def test_find_max(prices):
	max_num = strat.find_max(prices, 1)
	assert(max_num == 500)

	max_num = strat.find_max(prices, 2)
	assert(max_num == 499)
	
	max_num = strat.find_max(prices, 3)
	assert(max_num == 498)

	max_num = strat.find_max(prices, 51)
	assert(max_num == 450)

	max_num = strat.find_max(prices, 137)
	assert(max_num == 364)



def test_normalize(prices):
	deep_copy_prices = copy.deepcopy(prices)

	norm_prices = strat.normalize(deep_copy_prices, 1)
	assert(norm_prices[0][1] == 1/len(prices))
	assert(norm_prices[1][1] == 2/len(prices))
	assert(norm_prices[2][1] == 3/len(prices))
	assert(norm_prices[3][1] == 4/len(prices))
	assert(len(prices)/(len(prices)-1) >= norm_prices[-1][1] > len(prices)/(len(prices)+1))

	deep_copy_prices = copy.deepcopy(prices)

	norm_prices = strat.normalize(deep_copy_prices, 3)
	assert(1/(len(prices)-3) >= norm_prices[0][1] > 1/(len(prices)-1))
	assert(2/(len(prices)-3) >= norm_prices[1][1] > 2/(len(prices)-1))
	assert(len(prices)/(len(prices)-1) >= norm_prices[-3][1] > len(prices)/(len(prices)+1))



def test_calc_SMA(prices):
	SMA = strat.calc_SMA(prices, "350")
	assert(SMA["10"] == 495.5)
	assert(SMA["50"] == 475.5)
	assert(SMA["350"] == 325.5)

	SMA = strat.calc_SMA(prices, "250")
	assert(SMA["10"] == 495.5)
	assert(SMA["50"] == 475.5)
	assert(SMA["250"] == 375.5)

	SMA = strat.calc_SMA(prices, "350", 100)
	assert(SMA["10"] == 396.5)
	assert(SMA["50"] == 376.5)
	assert(SMA["350"] == 226.5)

	SMA = strat.calc_SMA(prices, "250", 100)
	assert(SMA["10"] == 396.5)
	assert(SMA["50"] == 376.5)
	assert(SMA["250"] == 276.5)


def test_handle_missing_data():
	# zero in the middle
	data = [['date', 1, 1, 1], 
			['date', 0, 0, 0],
			['date', 3, 3, 3]]
	data = pd.DataFrame(data)
	data = dpp.handle_missing_data(data)
	assert data.iloc[1, 1] == 2.0 and data.iloc[1, 2] == 2.0 and data.iloc[1, 3] == 2.0, "Zero in the middle test case failed."

	# consecutive zeros
	data = [['date', 1, 1, 1],
			['date', 0, 0, 0],
			['date', 0, 0, 0],
			['date', 3, 3, 3]]
	data = pd.DataFrame(data)
	data = dpp.handle_missing_data(data)
	assert data.iloc[1, 1] == 2.0 and data.iloc[1, 2] == 2.0 and data.iloc[1, 3] == 2.0, "Consecutive zeros test case failed."
	assert data.iloc[2, 1] == 2.5 and data.iloc[2, 2] == 2.5 and data.iloc[2, 3] == 2.5, "Consecutive zeros test case failed."

	# leading zero
	data = [['date', 0, 0, 0],
			['date', 1, 1, 1],
			['date', 0, 0, 0],
			['date', 3, 3, 3]]
	data = pd.DataFrame(data)
	data = dpp.handle_missing_data(data)
	assert data.iloc[0, 1] == 1.0 and data.iloc[0, 2] == 1.0 and data.iloc[0, 3] == 1.0, "Leading zero test case failed."

	# ending zero
	data = [['date', 1, 1, 1],
			['date', 1, 1, 1],
			['date', 2, 2, 2],
			['date', 0, 0, 0]]
	data = pd.DataFrame(data)
	data = dpp.handle_missing_data(data)
	assert data.iloc[3, 1] == 2.0 and data.iloc[3, 2] == 2.0 and data.iloc[3, 3] == 2.0, "Ending zero test case failed."


def test_normalize_data():
	assert False	


def test_calculate_SMAs():
	assert False


run_tests()

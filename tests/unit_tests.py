'''
RUN WITH: $ python3 -m unittest tests.unit_tests
'''
import src 
from src import investstrat as strat
import utils
from utils import data_preprocessor as dpp
from utils import data_processor as dp

import copy
import pandas as pd


def test_handle_missing_data():
	start_date = "2020-10-01"
	end_date = "2020-10-04"
	# missing dates
	data = [['2020-10-01', 1, 1, 1], 
			#comment out the following line to make test fail
			['2020-10-02', 0, 0, 0],
			['2020-10-03', 0, 0, 0],
			['2020-10-04', 3, 3, 3]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)

	# zero in the middle
	data = [['2020-10-01', 1, 1, 1], 
			['2020-10-02', 0, 0, 0],
			['2020-10-03', 3, 3, 3],
			['2020-10-04', 3, 3, 3]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)
	assert data.iloc[1, 1] == 2.0 and data.iloc[1, 2] == 2.0 and data.iloc[1, 3] == 2.0, "Zero in the middle test case failed."

	# consecutive zeros
	data = [['2020-10-01', 1, 1, 1],
			['2020-10-02', 0, 0, 0],
			['2020-10-03', 0, 0, 0],
			['2020-10-04', 3, 3, 3]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)
	assert data.iloc[1, 1] == 2.0 and data.iloc[1, 2] == 2.0 and data.iloc[1, 3] == 2.0, "Consecutive zeros test case failed."
	assert data.iloc[2, 1] == 2.5 and data.iloc[2, 2] == 2.5 and data.iloc[2, 3] == 2.5, "Consecutive zeros test case failed."

	# leading zero
	data = [['2020-10-01', 0, 0, 0],
			['2020-10-02', 1, 1, 1],
			['2020-10-03', 0, 0, 0],
			['2020-10-04', 3, 3, 3]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)
	assert data.iloc[0, 1] == 1.0 and data.iloc[0, 2] == 1.0 and data.iloc[0, 3] == 1.0, "Leading zero test case failed."

	# ending zero
	data = [['2020-10-01', 1, 1, 1],
			['2020-10-02', 1, 1, 1],
			['2020-10-03', 2, 2, 2],
			['2020-10-04', 0, 0, 0]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)
	assert data.iloc[3, 1] == 2.0 and data.iloc[3, 2] == 2.0 and data.iloc[3, 3] == 2.0, "Ending zero test case failed."


def test_normalize_data():
	data = [['2020-10-01', 1, 1, 1], 
			['2020-10-02', 0, 0, 0],
			['2020-10-03', 3, 3, 3],
			['2020-10-04', 3, 3, 3]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.normalize_data(data)
	assert data.iloc[0,1] == 1/3 and data.iloc[1,1] == 0 and data.iloc[2,1] == 1 and data.iloc[3,1] == 1, "Normalize data test failed."


def test_calculate_price_SMAs():
	data = [['x', 0]]
	for i in range(1,500):
		data.append(['x', data[i-1][1]+1])
	data = pd.DataFrame(data, columns=["date", "price"])
	data = dpp.calculate_price_SMAs(data)
	'''
	(498+497+496+495+494)/5 == data.iloc[5,2]
	Must start at 6th item from the last in order to be able to subtract
	'''
	assert (498+497+496+495+494)/5 == data.iloc[5,2]


def test_calculate_fear_greed_SMAs():
	assert False



def run_data_aggregator_tests():
	pass



def run_data_preprocessor_tests():
	test_handle_missing_data()
	print("test_handle_missing_data() tests all passed")
	test_normalize_data()
	print("test_normalize_data() tests all passed")
	test_calculate_price_SMAs()
	print("test_calculate_price_SMAs() tests all passed")
	test_calculate_fear_greed_SMAs()
	print("test_calculate_fear_greed_SMAs() tests all passed")



def run_data_processor_tests():
	test_generate_dataset()
	print("test_generate_dataset() tests all passed")
	test_get_datasets()
	print("test_get_datasets() tests all passed")
	test_shuffle_data()
	print("test_shuffle_data() tests all passed")



def run_tests():
	run_data_aggregator_tests()
	run_data_preprocessor_tests()
	#run_data_processor_tests()
	


run_tests()

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

#
# ---------- DATA AGGREGATOR TESTS ----------
#

#
# ---------- DATA PREPROCESSOR TESTS ----------
#
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
	# reverses dataframe as it does in the calculate_SMAs() method in the data_preprocessor.py file that calls the method being tested
	data = data.reindex(index=data.index[::-1]).reset_index()
	data = data.drop(columns=["index"])

	data = dpp.calculate_price_SMAs(data)
	'''
	(498+497+496+495+494)/5 == data.iloc[5,2]
	Must start at 6th item from the last in order to be able to subtract the first item added to the moving total
	'''
	total = lambda x,y: sum([n for n in range(x, y+1)]) 
	assert total(494,498)/5 == data.iloc[5,2], "The 5-day price SMA test failed."
	assert total(489,498)/10 == data.iloc[10,3], "The 10-day price SMA test failed."
	assert total(474,498)/25 == data.iloc[25,4], "The 25-day price SMA test failed."
	assert total(449,498)/50 == data.iloc[50,5], "The 50-day price SMA test failed."
	assert total(424,498)/75 == data.iloc[75,6], "The 75-day price SMA test failed."
	assert total(399,498)/100 == data.iloc[100,7], "The 100-day price SMA test failed."
	assert total(349,498)/150 == data.iloc[150,8], "The 150-day price SMA test failed."
	assert total(299,498)/200 == data.iloc[200,9], "The 200-day price SMA test failed."
	assert total(249,498)/250 == data.iloc[250,10], "The 250-day price SMA test failed."
	assert total(199,498)/300 == data.iloc[300,11], "The 300-day price SMA test failed."
	assert total(149,498)/350 == data.iloc[350,12], "The 350-day price SMA test failed."



def test_calculate_fear_greed_SMAs():
	data = [['x', 0]]
	for i in range(1,500):
		data.append(['x', data[i-1][1]+1])
	data = pd.DataFrame(data, columns=["date", "fear_greed"])
	# reverses dataframe as it does in the calculate_SMAs() method in the data_preprocessor.py file that calls the method being tested
	data = data.reindex(index=data.index[::-1]).reset_index()
	data = data.drop(columns=["index"])

	data = dpp.calculate_fear_greed_SMAs(data)
	'''
	(498+497+496)/3 == data.iloc[3,2]
	Must start at 4th item from the last in order to be able to subtract the first item added to the moving total
	'''
	total = lambda x,y: sum([n for n in range(x, y+1)]) 
	assert total(496,498)/3 == data.iloc[3,2], "The 3-day fear/greed SMA test failed."
	assert total(494,498)/5 == data.iloc[5,3], "The 5-day fear/greed SMA test failed."
	assert total(492,498)/7 == data.iloc[7,4], "The 7-day fear/greed SMA test failed."
	assert total(490,498)/9 == data.iloc[9,5], "The 9-day fear/greed SMA test failed."
	assert total(488,498)/11 == data.iloc[11,6], "The 11-day fear/greed SMA test failed."
	assert total(486,498)/13 == data.iloc[13,7], "The 13-day fear/greed SMA test failed."
	assert total(484,498)/15 == data.iloc[15,8], "The 15-day fear/greed SMA test failed."
	assert total(469,498)/30 == data.iloc[30,9], "The 30-day fear/greed SMA test failed."


#
# ---------- DATA PROCESSOR TESTS ----------
#
def test_generate_dataset():
	data = []
	for _ in range(7):
		row = [x+1 for x in range(26)]
		data.append(row)
	data[0].append(0)
	data[1].append(1)
	data[2].append(2)
	data[3].append(2)
	data[4].append(3)
	data[5].append(3)
	data[6].append(3)
	data = pd.DataFrame(data, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, "signal"])

	# No augmentation
	altered_data = dp.generate_dataset(data, len(data), 0)
	assert len(altered_data) == 7 and len(altered_data[0]) == 2 and len(altered_data[0][0]) == 26, "No augmentation test failed."

	# 10x augmentation
	altered_data = dp.generate_dataset(data, len(data), 0, 10)
	# 10*3 for signal 0
	# 10*3 for signal 1
	# 10*2 for signal 2
	# 10*2 for signal 2
	# 10*1 for signal 3
	# 10*1 for signal 3
	# 10*1 for signal 3
	# + 7 for the original elements in the data
	assert len(altered_data) == (10*3)+(10*3)+(10*2)+(10*2)+(10*1)+(10*1)+(10*1)+7 and len(altered_data[0]) == 2 and len(altered_data[0][0]) == 26, "Augmentation test failed."

	# Testing offset
	altered_data = dp.generate_dataset(data, len(data), 2, 10)
	assert len(altered_data) == (10*2)+(10*2)+(10*1)+(10*1)+(10*1)+5 and len(altered_data[0]) == 2 and len(altered_data[0][0]) == 26, "Offset test failed."
	
	# Testing limit
	altered_data = dp.generate_dataset(data, len(data)-2, 0, 10)
	assert len(altered_data) == (10*2)+(10*2)+(10*1)+(10*1)+(10*2)+5 and len(altered_data[0]) == 2 and len(altered_data[0][0]) == 26, "Limit test failed."




def test_get_datasets():
	assert False



def test_shuffle_data():
	assert False



#
# ---------- GENERAL METHODS ----------
#

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
	run_data_processor_tests()
	


run_tests()

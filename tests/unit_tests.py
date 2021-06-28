'''
RUN WITH: $ python3 -m unittest tests.unit_tests
'''
import utils
from utils import data_aggregator as da
from utils import data_preprocessor as dpp
from utils import data_processor as dp
from utils import neural_nets as nn

import os
import copy
import pandas as pd
import numpy as np
from datetime import date, datetime

#
# ---------- DATA AGGREGATOR TESTS ----------
#
def test_get_correct_date_format():
	wonky_date_str = "2021-06-14" 
	wonky_date = datetime.strptime(wonky_date_str, '%Y-%m-%d')
	correct_date = da.get_correct_date_format(wonky_date)

	assert str(correct_date) == "14-06-2021", "Failed get_correct_date_format test."



def test_get_community_score():
	pass



def test_get_dev_score():
	pass



def test_get_public_interest_score():
	pass



def test_extract_basic_data():
	pass



def test_get_time():
	pass

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

	# date type
	assert type(data.iloc[0, 0]) == pd._libs.tslibs.timestamps.Timestamp and type(data.iloc[1, 0]) == pd._libs.tslibs.timestamps.Timestamp and type(data.iloc[2, 0]) == pd._libs.tslibs.timestamps.Timestamp and type(data.iloc[3, 0]) == pd._libs.tslibs.timestamps.Timestamp, "Date column type changed test failed"


	# zero in the middle
	data = [['2020-10-01', 1, 1, 1], 
			['2020-10-02', 0, 0, 0],
			['2020-10-03', 3, 3, 3],
			['2020-10-04', 3, 3, 3]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)

	assert data.iloc[0, 1] == 1.0, "Zero in the middle (first row, second column) test failed."
	assert data.iloc[0, 2] == 1.0, "Zero in the middle (first row, third column) test failed."
	assert data.iloc[0, 3] == 1.0, "Zero in the middle (first row, final column) test failed."

	assert data.iloc[1, 1] == 2.0, "Zero in the middle (second row, second column) test failed."
	assert data.iloc[1, 2] == 2.0, "Zero in the middle (second row, third column) test failed."
	assert data.iloc[1, 3] == 2.0, "Zero in the middle (second row, final column) test failed."

	assert data.iloc[2, 1] == 3.0, "Zero in the middle (third row, second column) test failed."
	assert data.iloc[2, 2] == 3.0, "Zero in the middle (third row, third column) test failed."
	assert data.iloc[2, 3] == 3.0, "Zero in the middle (third row, final column) test failed."

	assert data.iloc[3, 1] == 3.0, "Zero in the middle (final row, second column) test failed."
	assert data.iloc[3, 2] == 3.0, "Zero in the middle (final row, third column) test failed."
	assert data.iloc[3, 3] == 3.0, "Zero in the middle (final row, final column) test failed."


	# consecutive zeros
	data = [['2020-10-01', 1, 1, 1],
			['2020-10-02', 0, 0, 0],
			['2020-10-03', 0, 0, 0],
			['2020-10-04', 3, 3, 3]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)
	
	assert data.iloc[0, 1] == 1.0, "Consecutive zeros (first row, second column) test case failed."
	assert data.iloc[0, 2] == 1.0, "Consecutive zeros (first row, third column) test case failed."
	assert data.iloc[0, 3] == 1.0, "Consecutive zeros (first row, final column) test case failed."

	assert data.iloc[1, 1] == 2.0, "Consecutive zeros (second row, second column) test case failed."
	assert data.iloc[1, 2] == 2.0, "Consecutive zeros (second row, third column) test case failed."
	assert data.iloc[1, 3] == 2.0, "Consecutive zeros (second row, final column) test case failed."
	
	assert data.iloc[2, 1] == 2.5, "Consecutive zeros (third row, second column) test case failed."
	assert data.iloc[2, 2] == 2.5, "Consecutive zeros (third row, third column) test case failed."
	assert data.iloc[2, 3] == 2.5, "Consecutive zeros (third row, final column) test case failed."

	assert data.iloc[3, 1] == 3.0, "Consecutive zeros (final row, second column) test case failed."
	assert data.iloc[3, 2] == 3.0, "Consecutive zeros (final row, third column) test case failed."
	assert data.iloc[3, 3] == 3.0, "Consecutive zeros (final row, final column) test case failed."


	# leading zero
	data = [['2020-10-01', 0, 0, 0],
			['2020-10-02', 1, 1, 1],
			['2020-10-03', 0, 0, 0],
			['2020-10-04', 3, 3, 3]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)

	assert data.iloc[0, 1] == 1.0, "Leading zero (first row, second column) test failed."
	assert data.iloc[0, 2] == 1.0, "Leading zero (first row, third column) test failed."
	assert data.iloc[0, 3] == 1.0, "Leading zero (first row, final column) test failed."

	assert data.iloc[1, 1] == 1.0, "Leading zero (second row, second column) test failed."
	assert data.iloc[1, 2] == 1.0, "Leading zero (second row, third column) test failed."
	assert data.iloc[1, 3] == 1.0, "Leading zero (second row, final column) test failed."

	assert data.iloc[2, 1] == 2.0, "Leading zero (third row, second column) test failed."
	assert data.iloc[2, 2] == 2.0, "Leading zero (third row, third column) test failed."
	assert data.iloc[2, 3] == 2.0, "Leading zero (third row, final column) test failed."

	assert data.iloc[3, 1] == 3.0, "Leading zero (final row, second column) test failed."
	assert data.iloc[3, 2] == 3.0, "Leading zero (final row, third column) test failed."
	assert data.iloc[3, 3] == 3.0, "Leading zero (final row, final column) test failed."


	# ending zero
	data = [['2020-10-01', 1, 1, 1],
			['2020-10-02', 1, 1, 1],
			['2020-10-03', 2, 2, 2],
			['2020-10-04', 0, 0, 0]]
	data = pd.DataFrame(data, columns=["date", 1, 2, 3])
	data = dpp.handle_missing_data(data, start_date, end_date)

	assert data.iloc[0, 1] == 1.0, "Ending zero (first row, second column) test failed."
	assert data.iloc[0, 2] == 1.0, "Ending zero (first row, third column) test failed."
	assert data.iloc[0, 3] == 1.0, "Ending zero (first row, final column) test failed."

	assert data.iloc[1, 1] == 1.0, "Ending zero (second row, second column) test failed."
	assert data.iloc[1, 2] == 1.0, "Ending zero (second row, third column) test failed."
	assert data.iloc[1, 3] == 1.0, "Ending zero (second row, final column) test failed."

	assert data.iloc[2, 1] == 2.0, "Ending zero (third row, second column) test failed."
	assert data.iloc[2, 2] == 2.0, "Ending zero (third row, third column) test failed."
	assert data.iloc[2, 3] == 2.0, "Ending zero (third row, final column) test failed."

	assert data.iloc[3, 1] == 2.0, "Ending zero (final row, second column) test failed."
	assert data.iloc[3, 2] == 2.0, "Ending zero (final row, third column) test failed."
	assert data.iloc[3, 3] == 2.0, "Ending zero (final row, final column) test failed."



def test_normalize_data():
	# check fear greed are out of 100
	data = [['2020-10-01', 1, 1, 1], 
			['2020-10-02', 1, 1, 1],
			['2020-10-03', 4, 4, 4],
			['2020-10-04', 2, 2, 2]]
	data = pd.DataFrame(data, columns=["date", "fear_greed", 2, 3])
	data = dpp.normalize_data(data)
	
	assert data.iloc[0,1] == 1/100, "Normalize data (first row) fear_greed index test failed."
	assert data.iloc[1,1] == 1/100, "Normalize data (second row) fear_greed index test failed."
	assert data.iloc[2,1] == 4/100, "Normalize data (third row) fear_greed index test failed."
	assert data.iloc[3,1] == 2/100, "Normalize data (final row) fear_greed index test failed."


	# standard
	data = [['2020-10-01', 1, 1, 1], 
			['2020-10-02', 1, 1, 1],
			['2020-10-03', 4, 4, 4],
			['2020-10-04', 2, 2, 2]]
	data = pd.DataFrame(data, columns=["date", "fear_greed", 2, 3])
	data = dpp.normalize_data(data)

	assert data.iloc[0,2] == 1, "Normalize data (first row) standard test failed."
	assert data.iloc[1,2] == 1, "Normalize data (second row) standard test failed." 
	assert data.iloc[2,2] == 1, "Normalize data (third row) standard test failed." 
	assert data.iloc[3,2] == 1/3, "Normalize data (final row) standard test failed."


	# with zeros
	data = [['2020-10-01', 0, 0, 0], 
			['2020-10-02', 1, 1, 1],
			['2020-10-03', 4, 4, 4],
			['2020-10-04', 2, 2, 2]]
	data = pd.DataFrame(data, columns=["date", "fear_greed", 2, 3])
	data = dpp.normalize_data(data)

	assert data.iloc[0,2] == 1, "Normalize data (first row) w/ zeros test failed." 
	assert data.iloc[1,2] == 1, "Normalize data (second row) w/ zeros test failed." 
	assert data.iloc[2,2] == 1, "Normalize data (third row) w/ zeros test failed." 
	assert data.iloc[3,2] == 1/2, "Normalize data (final row) w/ zeros test failed."


	# with only zeros
	data = [['2020-10-01', 0, 0, 0], 
			['2020-10-02', 0, 0, 0],
			['2020-10-03', 0, 0, 0],
			['2020-10-04', 0, 0, 0]]
	data = pd.DataFrame(data, columns=["date", "fear_greed", 2, 3])
	data = dpp.normalize_data(data)
	assert data.iloc[0,2] == 1, "Normalize data (first row) w/ only zeros test failed." 
	assert data.iloc[1,2] == 0, "Normalize data (second row) w/ only zeros test failed." 
	assert data.iloc[2,2] == 0, "Normalize data (third row) w/ only zeros test failed." 
	assert data.iloc[3,2] == 0, "Normalize data (final row) w/ only zeros test failed."



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
	# create a communal dataset to use for all tests in this section
	data = []
	for _ in range(7):
		row = [x+1 for x in range(nn.N_FEATURES)]
		data.append(row)
	data[0].append(0)
	data[1].append(1)
	data[2].append(2)
	data[3].append(2)
	data[4].append(3)
	data[5].append(3)
	data[6].append(3)
	data = pd.DataFrame(data, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, "signal"])

	# No augmentation
	altered_data = dp.generate_dataset(data, len(data), 0)
	assert len(altered_data) == 7, "No augmentation dataset length test failed"
	assert len(altered_data[0]) == 2, "No augmentation feature/target tuple length test failed"
	assert len(altered_data[0][0]) == nn.N_FEATURES, "No augmentation feature vector length test failed."

	# 10x augmentation
	# 10*3 for signal 0
	# 10*3 for signal 1
	# 10*2 for signal 2
	# 10*2 for signal 2
	# 10*1 for signal 3
	# 10*1 for signal 3
	# 10*1 for signal 3
	# + 7 for the original elements in the data
	altered_data = dp.generate_dataset(data, len(data), 0, 10)
	assert len(altered_data) == (10*3)+(10*3)+(10*2)+(10*2)+(10*1)+(10*1)+(10*1)+7, "10x augmentation dataset length test failed"
	assert len(altered_data[0]) == 2, "10x augmentation feature/target tuple length test failed" 
	assert len(altered_data[0][0]) == nn.N_FEATURES, "10x augmentation feature vector length test failed."

	# Testing offset
	altered_data = dp.generate_dataset(data, len(data), 2, 10)
	assert len(altered_data) == (10*2)+(10*2)+(10*1)+(10*1)+(10*1)+5, "2 offset dataset length test failed"
	assert len(altered_data[0]) == 2, "2 offset feature/target tuple length test failed"  
	assert len(altered_data[0][0]) == nn.N_FEATURES, "2 offset feature vector length test failed."
	
	# Testing limit
	altered_data = dp.generate_dataset(data, len(data)-2, 0, 10)
	assert len(altered_data) == (10*2)+(10*2)+(10*1)+(10*1)+(10*2)+5, "2 limit dataset length test failed"
	assert len(altered_data[0]) == 2, "2 limit feature/target tuple length test failed"  
	assert len(altered_data[0][0]) == nn.N_FEATURES, "2 limit feature vector length test failed."



def create_fake_csv():
	data = []
	for i in range(100):
		dt_list = []
		for j in range(nn.N_FEATURES):
			dt_list.append(i)
		data.append(dt_list)

	col_labels = ["date"]
	for i in range(nn.N_FEATURES-2):
		col_labels.append(i)
	col_labels.append("signal")
	
	data = pd.DataFrame(data, columns=col_labels)
	data.to_csv("datasets/complete/fakecoin_historical_data_complete.csv")
	
	return data



def destroy_fake_coin():
	os.remove("datasets/complete/fakecoin_historical_data_complete.csv")



def test_get_datasets():
	coin = "fakecoin"
	data = create_fake_csv()
	train_data, valid_data, test_data = dp.get_datasets(coin, data_aug_factor=16)

	# test with standard 16x augmentation
	# len(data)*0.7*16 = the data augmentation portion
	# + (len(data)*0.7) = the original datapoints before augmentation
	assert len(train_data) == (len(data)*0.7*16) + (len(data)*0.7), "Failed train_data size test in get_datasets test."
	assert len(valid_data) == len(data)*0.15, "Failed valid_data size test in get_datasets test."
	assert len(test_data) == len(data)*0.15, "Failed test_data size test in get_datasets test."
	assert 68.9 < train_data[int(len(data)*0.7*16)+69][0][0] < 69.1, "Failed train_data value test in get_datasets test."
	assert valid_data[int(len(data)*0.15)-1][0][0] == 84, "Failed valid_data value test in get_datasets test."
	assert test_data[int(len(data)*0.15)-1][0][0] == 99, "Failed test_data value test in get_datasets test."

	# test with 0 augmentation
	train_data, valid_data, test_data = dp.get_datasets(coin)

	assert len(train_data) == len(data)*0.7, "Failed train_data size test in get_datasets test."
	assert len(valid_data) == len(data)*0.15, "Failed valid_data size test in get_datasets test."
	assert len(test_data) == len(data)*0.15, "Failed test_data size test in get_datasets test."
	assert train_data[int(len(data)*0.7)-1][0][0] == 69, "Failed train_data value test in get_datasets test."
	assert valid_data[int(len(data)*0.15)-1][0][0] == 84, "Failed valid_data value test in get_datasets test."
	assert test_data[int(len(data)*0.15)-1][0][0] == 99, "Failed test_data value test in get_datasets test."

	destroy_fake_coin()



def test_shuffle_data():
	data = [[0],
			[1],
			[2],
			[3],
			[4],
			[5],
			[6],
			[7],
			[8],
			[9]]
	
	data = dp.shuffle_data(data)

	assert (data[0] == [0] and data[1] == [1] and data[2] == [2] and data[3] == [3] and data[4] == [4] and data[5] == [5] and data[6] == [6] and data[7] == [7] and data[8] == [8] and data[9] == [9]) == False, "Failed random shuffling of data in shuffle_data test."



def test_terminate_early():
	prev_valid_losses = [ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0 ]
	assert dp.terminate_early(prev_valid_losses) == True, "Failed should terminate early in terminate_early test."

	prev_valid_losses.reverse()
	assert dp.terminate_early(prev_valid_losses) == False, "Failed should not terminate early in terminate_early test."



#
# ---------- GENERAL METHODS ----------
#

def run_data_aggregator_tests():
	test_get_correct_date_format()
	print("test_get_correct_date_format() tests all passed.")



def run_data_preprocessor_tests():
	test_handle_missing_data()
	print("test_handle_missing_data() tests all passed.")
	test_normalize_data()
	print("test_normalize_data() tests all passed.")
	test_calculate_price_SMAs()
	print("test_calculate_price_SMAs() tests all passed.")
	test_calculate_fear_greed_SMAs()
	print("test_calculate_fear_greed_SMAs() tests all passed.")



def run_data_processor_tests():
	test_generate_dataset()
	print("test_generate_dataset() tests all passed.")
	test_get_datasets()
	print("test_get_datasets() tests all passed.")
	test_shuffle_data()
	print("test_shuffle_data() tests all passed.")
	test_terminate_early()
	print("test_terminate_early() tests all passed.")



def run_tests():
	run_data_aggregator_tests()
	run_data_preprocessor_tests()
	run_data_processor_tests()
	


run_tests()

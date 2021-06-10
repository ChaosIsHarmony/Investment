import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import investstrat as strat
import copy


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



run_tests() 

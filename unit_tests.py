import investstrat as strat

def tests():
	prices = []
	for i in range(500):
		prices.append([i,float(i+1)])
	test_find_max(prices)
	test_normalize(prices)
	test_calc_SMA(prices)


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
	norm_prices = strat.normalize(prices, 1)
	assert(norm_prices[0][1] == 1/len(prices))
	assert(norm_prices[1][1] == 2/len(prices))
	assert(norm_prices[2][1] == 3/len(prices))
	assert(norm_prices[3][1] == 4/len(prices))
	assert(len(prices)/(len(prices)-1) >= norm_prices[-1][1] > len(prices)/(len(prices)+1))

	norm_prices = strat.normalize(prices, 3)
	assert(1/(len(prices)-3) >= norm_prices[0][1] > 1/(len(prices)-1))
	assert(2/(len(prices)-3) >= norm_prices[1][1] > 2/(len(prices)-1))
	assert(len(prices)/(len(prices)-1) >= norm_prices[-3][1] > len(prices)/(len(prices)+1))

def test_calc_SMA(prices):
	strat.calc_SMA(prices, 350, 1)
    
tests() 

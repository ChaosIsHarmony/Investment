import investstrat as strat 
from datetime import date, datetime, timedelta

prices_on = []

today = date.today()
print(today) 

coins = strat.coin_id
key = "200"

for coin in coins:
	print(coin)
	prices_orig = strat.get_chart(coin, "max")["prices"][1:]
	print(f"{len(prices_orig)} days of data")
	for i in range(len(prices_orig)):
		# replace missing values
		if prices_orig[i][1] == None:
			prices_orig[i][1] = (prices_orig[i-1][1] +prices_orig[i+1][1])/2
	days = len(prices_orig)
	signal_changes = {}
	prev_signal = ""

	for i in range(1, 15):#days-int(key)):
		SMA = {}
		prices_norm = strat.normalize(prices_orig, i)
		SMA = strat.calc_SMA(prices_norm, key, i)
		
		if key not in SMA.keys():
			break
		
		risk, d_risk, cur_max_ratio, decision = strat.generate_report(SMA, prices_norm, key, i)

		if decision != prev_signal:
			if i == 1:
				prev_signal = decision
			else:	
				signal_changes[i] = [f"{prev_signal} <- {decision}", risk, d_risk, cur_max_ratio]
				prev_signal = decision
	
	
	for days in signal_changes.keys():
		print(f"Date: {today-timedelta(days)} | Signal: {signal_changes[days][0]} | Risk Agg. Metrics: 50/{key} = {signal_changes[days][1]:.4f}; 10/50 = {signal_changes[days][2]:.4f}; Cur/Max = {signal_changes[days][3]:.4f}")

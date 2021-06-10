# used to import investstrat from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import investstrat as strat 
from datetime import date, datetime, timedelta

prices_on = []

today = date.today()
print(today) 

coins = strat.coin_id
key = "200"

days_ago = ""
while type(days_ago) == str:
	try:
		days_ago = int(input("How far back to look for signal change? (in days) "))
	except:
		print("Value must be an integer.")
		days_ago = ""

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

	for i in range(1, days_ago):#days-int(key)):
		SMA = {}
		prices_norm = strat.normalize(prices_orig, i)
		SMA = strat.calc_SMA(prices_norm, key, i)
		
		if key not in SMA.keys():
			break
		
		risk, d_risk_5, d_risk_10, d_risk_50, cur_max_ratio, decision = strat.generate_report(SMA, prices_norm, key, i)

		if decision != prev_signal:
			signal_changes[i] = [f"{prev_signal} <- {decision}", risk, d_risk_5, d_risk_10, d_risk_50, cur_max_ratio]
			prev_signal = decision
	
	
	for days in signal_changes.keys():
		print(f"\tDate: {today-timedelta(days)}") 
		print(f"\tSignal: {signal_changes[days][0]}")
		print("\tRisk Metrics:")
		print(f"\t\t50/{key} MA = {signal_changes[days][1]:.4f}")
		print(f"\t\t5 MA - 10 MA = {signal_changes[days][2]:.4f}")
		print(f"\t\t10 MA - 50 MA = {signal_changes[days][3]:.4f}")
		print(f"\t\t50 MA - 100 MA = {signal_changes[days][4]:.4f}")
		print(f"\t\tCur/Max = {signal_changes[days][5]:.4f}")

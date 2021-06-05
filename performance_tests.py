import investstrat as strat 
from datetime import date, datetime, timedelta

prices_on = []

today = date.today()
print(today) 

coins = ['bitcoin']

for coin in coins:
	signal_changes = strat.in_depth_report(coin, 350, 500)
	print(signal_changes)
	for days in signal_changes.keys():
		print(f"Date: {today-timedelta(days)} | Signal: {signal_changes[days][0]}")

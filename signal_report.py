import investstrat as strat


while (True):
	print("Available coins: ", strat.coin_id)
	coin = input("Which coin? ")

	# not an option
	if coin not in strat.coin_id and coin != "all":
		break

	key = input("What MA should the 50-day MA be compared to? (increments of 50) ")
	days = -1.0 # set as float so it enters loop
	while (type(days) != int):
		days = input("How far back? (in days) ")
		try:
			days = int(days)
		except:
			print("Days must be an integer value")

	
	# Report on all coins
	if coin == "all":
		for c in strat.coin_id:
			strat.most_recent_report(c, key, days)
		continue

	# In-depth report on single coin
	strat.in_depth_report(coin, key, days)

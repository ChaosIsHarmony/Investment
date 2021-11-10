btc_principle_cel = float(input("Starting BTC [cel]: "))
eth_principle_cel = float(input("Starting ETH [cel]: "))
btc_principle_nex = float(input("Starting BTC [nex]: "))
btc_apy_cel = 0.0625
eth_apy_cel = 0.0525
btc_apy_nex = 0.04
btc_daily_interest_cel = btc_apy_cel / 365
eth_daily_interest_cel = eth_apy_cel / 365
btc_daily_interest_nex = btc_apy_nex / 365
timeframe = int(input("Timeframe [weeks]: "))
dca = 100 / 65000

for week in range(timeframe):
    for day in range(7):
        btc_principle_cel *= 1 + btc_daily_interest_cel
        eth_principle_cel *= 1 + eth_daily_interest_cel
        btc_principle_nex *= 1 + btc_daily_interest_nex
    # add DCA
    btc_principle_cel += dca / 2
    btc_principle_nex + dca / 2
    dca *= 0.99
    if (week+1) % 52 == 0:
        dca *= 1.5

if timeframe < 53:
    btc_value_usd = 100000
    eth_value_usd = 10000
    sol_value_usd = 750
elif timeframe < 105:
    btc_value_usd = 200000
    eth_value_usd = 20000
    sol_value_usd = 1500
elif timeframe < 157:
    btc_value_usd = 300000
    eth_value_usd = 30000
    sol_value_usd = 2500
elif timeframe < 209:
    btc_value_usd = 400000
    eth_value_usd = 35000
    sol_value_usd = 3000





total_btc = btc_principle_cel + btc_principle_nex + 0.037
total_eth = eth_principle_cel + 0.49
total_sol = 20
total_value_usd = (total_btc * btc_value_usd) + (total_eth * eth_value_usd) + (total_sol * sol_value_usd)
usd_twd = 27

print(f"Total btc principle after week {timeframe} = {btc_principle_cel + btc_principle_nex:0.5f}")
print(f"Total eth principle after week {timeframe} = {eth_principle_cel:0.5f}")
print(f"Total usd principle after week {timeframe} = {total_value_usd:0.2f}")
print(f"Total twd principle after week {timeframe} = {total_value_usd * usd_twd:0.0f}")



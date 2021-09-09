import pandas as pd
import numpy as np
import scipy.stats
import random
from typing import List
from . import common


def get_datasets(coins: List[str], interval: int) -> pd.DataFrame:
    portfolio_dataset = pd.DataFrame()
    for coin in coins:
        data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")
        data = data.iloc[:interval, :] # trim to specified interval

        start_date = data.iloc[0,0]
        end_date = data.iloc[data.shape[0]-1,0]

        data = data.drop(columns=["market_cap", "volume", "fear_greed"])
        data = common.handle_missing_data(coin, data, start_date, end_date)
        data = data.iloc[::-1] # reverse dataset for later pct_change calculations

        portfolio_dataset[coin+"_price"] = data["price"]

    return portfolio_dataset



def calculate_new_subtotals(subtotals: List[float], returns: List[float], year: int) -> List[float]:
    n_cols = len(subtotals)
    # divide by year to account for diminishing returns/volatility
    pct_changes = [(1 + random.choice(returns.iloc[1:,col]))/year for col in range(n_cols)]
    new_subtotals = [(subtotals[col] * pct_changes[col]) for col in range(n_cols)]
    return new_subtotals




coins = ["bitcoin", "cardano", "ethereum", "solana"]
start_pct = [0.45, 0.2, 0.15, 0.2]
interval = 365
portfolio_dataset = get_datasets(coins, interval)
returns = portfolio_dataset.pct_change()
np.random.seed(42)
sum_of_trials = 0
portfolio_start_amt = int(input("Portfolio start amount: "))
dca_amt = int(input("DCA amount: "))
n_years = int(input("Number of years: "))

for trial in range(1, 10000):
    trial_total = 0
    start_total = portfolio_start_amt
    subtotals = [start_total*start_pct[0], start_total*start_pct[1], start_total*start_pct[2], start_total*start_pct[3]]
    for year in range(1,n_years+1):
        for day in range(1,interval):
            subtotals = calculate_new_subtotals(subtotals, returns, year)
            if day % 7 == 0:
                dca_coin_ind = random.choice([0,1,2,3])
                subtotals[dca_coin_ind] += dca_amt
        # interest earned from staking
        subtotals[1] *= 1.05
        subtotals[3] *= 1.05
        trial_total += sum(subtotals)
        print(f"Year: {year} | Total: {trial_total}")
    sum_of_trials += trial_total
    print(f"Moving average: {sum_of_trials/trial:,.0f}")


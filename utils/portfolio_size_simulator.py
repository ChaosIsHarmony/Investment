import pandas as pd
import numpy as np
import scipy.stats
import random
from typing import List
from . import common


def custom_distribution(min_val, max_val, mean, std):
    scale = max_val - min_val
    location = min_val
    # Mean and standard deviation of the unscaled beta distribution
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    # Computation of alpha and beta can be derived from mean and variance formulas
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    # Not all parameters may produce a valid distribution
    if alpha <= 0 or beta <= 0:
        raise ValueError('Cannot create distribution for the given parameters.')
    # Make scaled beta distribution with computed parameters
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)



def get_datasets(coins: List[str], interval: int) -> pd.DataFrame:
    portfolio_dataset = pd.DataFrame()
    for coin in coins:
        data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")
        data = data.iloc[:interval, :] # trim to specified interval

        start_date = data.iloc[0,0]
        end_date = data.iloc[data.shape[0]-1,0]

        data = data.drop(columns=["market_cap", "volume", "fear_greed"])
        data = common.handle_missing_data(data, start_date, end_date)
        data = data.iloc[::-1] # reverse dataset for later pct_change calculations

        portfolio_dataset[coin+"_price"] = data["price"]

    return portfolio_dataset


coins = ["bitcoin", "cardano", "ethereum", "solana"]
interval = 365
portfolio_dataset = get_datasets(coins, interval)
returns = portfolio_dataset.pct_change()

np.random.seed(42)
min_vals = [-0.136994, -0.248368, -0.265229, -0.365443]
max_vals = [0.188653, 0.315895, 0.247752, 0.471329]
means = [0.004453, 0.009788, 0.006958, 0.011337]
stds = [0.040425, 0.069280, 0.055404, 0.093421]


sum_of_trials = 0
for trial in range(1, 10000):
    trial_total = 0
    start_total = 300000
    subtotals = [start_total*0.45, start_total*0.2, start_total*0.15, start_total*0.2]
    dca_amt = 30000
    for year in range(1,6):
        #  distributions = [custom_distribution(min_vals[i], max_vals[i], means[i], stds[i]) for i in range(len(coins))]
        #  samples = [distributions[i].rvs(size=365) for i in range(len(coins))]
        for day in range(1,interval):
            pct_changes = [(returns.iloc[day,i]*random.uniform(-1.5,2.0)) for i in range(len(subtotals))]
            subtotals = [subtotals[i] * (1 + (pct_changes[i]/year**(3/2))) for i in range(len(subtotals))]
            if day % 60 == 0:
                subtotals = [subtotals[0] + (dca_amt*0.45),
                            subtotals[1] + (dca_amt*0.2),
                            subtotals[2] + (dca_amt*0.15),
                            subtotals[3] + (dca_amt*0.2)]
        trial_total += sum(subtotals)
    sum_of_trials += trial_total
    print(f"Moving average: {sum_of_trials/trial:,.0f}")


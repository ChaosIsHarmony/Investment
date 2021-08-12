import concurrent.futures as cf
import joblib
import numpy as np
import pandas as pd
import torch
from . import common
from datetime import date, timedelta
from typing import List, Tuple


DECISIONS = ["BUY", "HODL", "SELL"]
PRICE = 0
MARKET_CAP = 1
VOLUME = 2
FEAR_GREED = 3
PRICE_5_SMA = 4
PRICE_10_SMA = PRICE_5_SMA+1
PRICE_25_SMA = PRICE_5_SMA+2
PRICE_50_SMA = PRICE_5_SMA+3
PRICE_75_SMA = PRICE_5_SMA+4
PRICE_100_SMA = PRICE_5_SMA+5
PRICE_150_SMA = PRICE_5_SMA+6
PRICE_200_SMA = PRICE_5_SMA+7
PRICE_250_SMA = PRICE_5_SMA+8
PRICE_300_SMA = PRICE_5_SMA+9
PRICE_350_SMA = PRICE_5_SMA+10
FG_3_SMA = 15
FG_5_SMA = FG_3_SMA+1
FG_7_SMA = FG_3_SMA+2
FG_9_SMA = FG_3_SMA+3
FG_11_SMA = FG_3_SMA+4
FG_13_SMA = FG_3_SMA+5
FG_15_SMA = FG_3_SMA+6
FG_30_SMA = FG_3_SMA+7



def fetch_new_data(n_days: int) -> None:
    '''
    Uses multithreading to speed up fetching process.
    '''
    with cf.ThreadPoolExecutor() as executor:
        results = [executor.submit(common.aggregate_new_data, coin, n_days) for coin in common.coins]
        for thread in cf.as_completed(results):
            print(thread.result())

    with cf.ThreadPoolExecutor() as executor:
        results = [executor.submit(common.merge_newly_aggregated_data, coin) for coin in common.coins]
        for thread in cf.as_completed(results):
            print(thread.result())



def process_individual_coin_new_data(start_date: str, end_date: str, coin: str) -> None:
    data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw.csv")
    data = common.clean_coin_data(coin, data, start_date, end_date)
    data.to_csv(f"datasets/clean/{coin}_historical_data_clean.csv", index=False)

    return f"All new data cleaned for {coin}."



def process_new_data() -> None:
    '''
    Uses multiprocessing to speed up data processing.
    '''
    start_date = str(date.today())
    end_date = "2020-08-23" #the first day of data of: the youngest asset: polkadot
    with cf.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_individual_coin_new_data, start_date, end_date, coin) for coin in common.coins]
        for process in cf.as_completed(results):
            print(process.result())



def get_fg_indicator(fg_index: float) -> str:
    if fg_index < 0.2:
        return "Extreme Fear"
    elif fg_index < 0.4:
        return "Fear"
    elif fg_index < 0.6:
        return "Neutral"
    elif fg_index < 0.8:
        return "Greed"
    else:
        return "Extreme Greed"



def populate_stat_report_full(coin: str, data: pd.DataFrame, raw_data: pd.DataFrame, report: List[str]) -> None:
    basic_stats = ["\n\n\n________________________________________",
                   f"Report for {coin.upper()}:",
                   "Basic Stats",
                   "[1.0 is the highest; 0.0 is the lowest]",
                   f"price:\t\t\t{data[PRICE]:.6f}",
                   f"market_cap:\t\t{data[MARKET_CAP]:.6f}",
                   f"volume:\t\t\t{data[VOLUME]:.6f}",
                   f"fear/greed:\t\t{data[FEAR_GREED]:.6f} [{get_fg_indicator(data[FEAR_GREED])}]",
                   f"sharpe_ratio:\t{common.get_sharpe_ratio(coin):.6f}",
                   f"UPI:\t\t\t{common.get_upi(coin):.6f}"]

    price_ratios = ["\nPrice Ratios",
                    "[>0 means greater risk/overvalued; <0 means less risk/undervalued]",
                    f"5-day/10-day:\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_10_SMA]:>9.6f}",
                    f"10-day/25-day:\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_25_SMA]:>9.6f}",
                    f"25-day/50-day:\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_50_SMA]:>9.6f}"]

    if raw_data[PRICE_200_SMA] > 0:
        price_ratios.append("x-day/200-day")
        price_ratios.append(f"\t5-day/200-day:\t\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
        price_ratios.append(f"\t10-day/200-day:\t\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
        price_ratios.append(f"\t25-day/200-day:\t\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
        price_ratios.append(f"\t50-day/200-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
        price_ratios.append(f"\t75-day/200-day:\t\t{raw_data[PRICE_75_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
        price_ratios.append(f"\t100-day/200-day:\t{raw_data[PRICE_100_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
        price_ratios.append(f"\t150-day/200-day:\t{raw_data[PRICE_150_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
    if raw_data[PRICE_250_SMA] > 0:
        price_ratios.append("x-day/250-day")
        price_ratios.append(f"\t5-day/250-day:\t\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
        price_ratios.append(f"\t10-day/250-day:\t\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
        price_ratios.append(f"\t25-day/250-day:\t\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
        price_ratios.append(f"\t50-day/250-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
        price_ratios.append(f"\t75-day/250-day:\t\t{raw_data[PRICE_75_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
        price_ratios.append(f"\t100-day/250-day:\t{raw_data[PRICE_100_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
        price_ratios.append(f"\t150-day/250-day:\t{raw_data[PRICE_150_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
    if raw_data[PRICE_300_SMA] > 0:
        price_ratios.append("x-day/300-day")
        price_ratios.append(f"\t5-day/300-day:\t\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
        price_ratios.append(f"\t10-day/300-day:\t\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
        price_ratios.append(f"\t25-day/300-day:\t\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
        price_ratios.append(f"\t50-day/300-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
        price_ratios.append(f"\t75-day/300-day:\t\t{raw_data[PRICE_75_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
        price_ratios.append(f"\t100-day/300-day:\t{raw_data[PRICE_100_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
        price_ratios.append(f"\t150-day/300-day:\t{raw_data[PRICE_150_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
    if raw_data[PRICE_350_SMA] > 0:
        price_ratios.append("x-day/350-day")
        price_ratios.append(f"\t5-day/350-day:\t\t{raw_data[PRICE_5_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
        price_ratios.append(f"\t10-day/350-day:\t\t{raw_data[PRICE_10_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
        price_ratios.append(f"\t25-day/350-day:\t\t{raw_data[PRICE_25_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
        price_ratios.append(f"\t50-day/350-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
        price_ratios.append(f"\t75-day/350-day:\t\t{raw_data[PRICE_75_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
        price_ratios.append(f"\t100-day/350-day:\t{raw_data[PRICE_100_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
        price_ratios.append(f"\t150-day/350-day:\t{raw_data[PRICE_150_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
    else:
        price_ratios.append("WARNING: DATA MISSING FROM SMAs; MODEL MAY BE UNRELIABLE")

    price_deltas = ["\nPrice Deltas",
                    "[<0 shows a decrease; >0 shows an increase]",
                    f"5-day -> Present:\t\t{data[PRICE]-data[PRICE_5_SMA]:>9.6f}",
                    f"10-day -> 5-day:\t\t{data[PRICE_5_SMA]-data[PRICE_10_SMA]:>9.6f}",
                    f"25-day -> 10-day:\t\t{data[PRICE_10_SMA]-data[PRICE_25_SMA]:>9.6f}",
                    f"50-day -> 25-day:\t\t{data[PRICE_25_SMA]-data[PRICE_50_SMA]:>9.6f}",
                    f"75-day -> 50-day:\t\t{data[PRICE_50_SMA]-data[PRICE_75_SMA]:>9.6f}",
                    f"100-day -> 75-day:\t\t{data[PRICE_75_SMA]-data[PRICE_100_SMA]:>9.6f}"]

    fear_greed_deltas = ["\nFear/Greed Deltas",
                         "[>0 is greedier; <0 is more fearful]",
                         f"3-day -> Present:\t{data[FEAR_GREED]-data[FG_3_SMA]:>9.6f}",
                         f"5-day -> 3-day:\t\t{data[FG_3_SMA]-data[FG_5_SMA]:>9.6f}",
                         f"7-day -> 5-day\t\t{data[FG_5_SMA]-data[FG_7_SMA]:>9.6f}",
                         f"9-day -> 7-day:\t\t{data[FG_7_SMA]-data[FG_9_SMA]:>9.6f}",
                         f"11-day -> 9-day:\t{data[FG_9_SMA]-data[FG_11_SMA]:>9.6f}",
                         f"13-day -> 11-day:\t{data[FG_11_SMA]-data[FG_13_SMA]:>9.6f}",
                         f"15-day -> 13-day:\t{data[FG_13_SMA]-data[FG_15_SMA]:>9.6f}",
                         f"30-day -> 15-day:\t{data[FG_15_SMA]-data[FG_30_SMA]:>9.6f}"]

    for item in basic_stats:
        report.append(item)
    for item in price_ratios:
        report.append(item)
    for item in price_deltas:
        report.append(item)
    for item in fear_greed_deltas:
        report.append(item)



def populate_stat_report_essentials(coin: str, data: pd.DataFrame, raw_data: pd.DataFrame, report: List[str]) -> None:
    basic_stats = ["\n\n\n________________________________________",
                   f"Report for {coin.upper()}:",
                   "Basic Stats",
                   "[1.0 is the highest; 0.0 is the lowest]",
                   f"price:\t\t\t{data[PRICE]:.6f}",
                   f"market_cap:\t\t{data[MARKET_CAP]:.6f}",
                   f"volume:\t\t\t{data[VOLUME]:.6f}",
                   f"fear/greed:\t\t{data[FEAR_GREED]:.6f} [{get_fg_indicator(data[FEAR_GREED])}]",
                   f"sharpe_ratio:\t{common.get_sharpe_ratio(coin):.6f}",
                   f"UPI:\t\t\t{common.get_upi(coin):.6f}"]

    price_ratios = ["\nPrice Ratios",
                    "[>0 means greater risk/overvalued; <0 means less risk/undervalued]"]

    if raw_data[PRICE_150_SMA] > 0:
        price_ratios.append(f"50-day/150-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_150_SMA]:>9.6f}")
    if raw_data[PRICE_200_SMA] > 0:
        price_ratios.append(f"50-day/200-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_200_SMA]:>9.6f}")
    if raw_data[PRICE_250_SMA] > 0:
        price_ratios.append(f"50-day/250-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_250_SMA]:>9.6f}")
    if raw_data[PRICE_300_SMA] > 0:
        price_ratios.append(f"50-day/300-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_300_SMA]:>9.6f}")
    if raw_data[PRICE_350_SMA] > 0:
        price_ratios.append(f"50-day/350-day:\t\t{raw_data[PRICE_50_SMA]/raw_data[PRICE_350_SMA]:>9.6f}")
    else:
        price_ratios.append("WARNING: DATA MISSING FROM SMAs; MODEL MAY BE UNRELIABLE")

    price_deltas = ["\nPrice Deltas",
                    "[<0 shows a decrease; >0 shows an increase]",
                    f"25-day -> Present:\t\t{data[PRICE]-data[PRICE_25_SMA]:>9.6f}",
                    f"50-day -> Present:\t\t{data[PRICE]-data[PRICE_50_SMA]:>9.6f}",
                    f"100-day -> Present:\t\t{data[PRICE]-data[PRICE_100_SMA]:>9.6f}"]

    fear_greed_deltas = ["\nFear/Greed Deltas",
                         "[>0 is greedier; <0 is more fearful]",
                         f"7-day -> Present:\t{data[FEAR_GREED]-data[FG_7_SMA]:>9.6f}",
                         f"15-day -> Present:\t{data[FEAR_GREED]-data[FG_15_SMA]:>9.6f}",
                         f"30-day -> Present:\t{data[FEAR_GREED]-data[FG_30_SMA]:>9.6f}"]

    for item in basic_stats:
        report.append(item)
    for item in price_ratios:
        report.append(item)
    for item in price_deltas:
        report.append(item)
    for item in fear_greed_deltas:
        report.append(item)



def get_models(best: List[str]) -> List[common.nn.CryptoSoothsayer]:
    models = []
    for i in range(len(best)):
        models.append(common.load_pretrained_model(best[i]))

    return models



def get_signal_strength(data: pd.DataFrame, raw_data: pd.DataFrame) -> Tuple[float, str]:
    # determine signal direction
    if raw_data[PRICE_200_SMA] > 0:
        ratio = raw_data[PRICE_50_SMA] / raw_data[PRICE_200_SMA]
        if ratio < 0.8:
            buy, hodl, sell = True, False, False
        elif ratio > 1.5:
            buy, hodl, sell = False, False, True
        else:
            buy, hodl, sell = False, True, False
    else:
        buy = True if data[FEAR_GREED] < 0.4 else False
        hodl = True if 0.4 <= data[FEAR_GREED] <= 0.75 else False
        sell = True if data[FEAR_GREED] > 0.75 else False


    signal_strength = 0
    if not hodl:
        # calculate how fearful/greedy market is
        if buy:
            signal_strength += ((0.5 - data[FEAR_GREED]) / 0.5)
        elif sell:
            signal_strength += ((data[FEAR_GREED] - 0.5) / 0.5)

        # calculate under/overvaluation
        tot_val = 0
        tot_metrics = 0
        if raw_data[PRICE_200_SMA] > 0:
            ratio = raw_data[PRICE_50_SMA] / raw_data[PRICE_200_SMA]
            if buy:
                tot_val += 1 / ratio
            elif sell:
                tot_val += ratio
            tot_metrics += 1

        if raw_data[PRICE_250_SMA] > 0:
            ratio = raw_data[PRICE_50_SMA] / raw_data[PRICE_250_SMA]
            if buy:
                tot_val += 1 / ratio
            elif sell:
                tot_val += ratio
            tot_metrics += 1

        if raw_data[PRICE_300_SMA] > 0:
            ratio = raw_data[PRICE_50_SMA] / raw_data[PRICE_300_SMA]
            if buy:
                tot_val += 1 / ratio
            elif sell:
                tot_val += ratio
            tot_metrics += 1

        if raw_data[PRICE_350_SMA] > 0:
            ratio = raw_data[PRICE_50_SMA] / raw_data[PRICE_350_SMA]
            if buy:
                tot_val += 1 / ratio
            elif sell:
                tot_val += ratio
            tot_metrics += 1

        signal_strength += 0 if tot_metrics == 0 else (tot_val / tot_metrics)

    # ultimate decision
    if buy:
        signal_direction = "BUY"
    elif sell:
        signal_direction = "SELL"
    else:
        signal_direction = "HODL"

    return signal_strength, signal_direction



def generate_signals(full_report: bool = False) -> List[str]:
    report = []
    best_models = []
    for coin in common.coins:
        # parse all asset-specific best-performing models OR just use the bitcoin one if no asset-specific ones exist (e.g., polkadot because it's too new)
        try:
            with open(f"reports/{coin}_best_performers.txt") as f:
                models = f.read().splitlines()
                for model in models:
                    best_models.append(model)
        except:
            with open("reports/all_best_performers.txt") as f:
                models = f.read().splitlines()
                for model in models:
                    best_models.append(model)

        # NOTE: raw_data is used for the SMA ratio calculations as the normalized data cannot adequately capture the ratios' significances
        data = pd.read_csv(f"datasets/clean/{coin}_historical_data_clean.csv")
        raw_data = pd.read_csv(f"datasets/raw/{coin}_historical_data_raw_all_features.csv")
        # extracts the most recent data as a python list
        data = data[data["date"] == str(date.today()-timedelta(0))].values.tolist()[0][1:-1]
        raw_data = raw_data[raw_data["date"] == str(date.today()-timedelta(0))].values.tolist()[0][1:-1]
        # stat report
        if full_report:
            populate_stat_report_full(coin, data, raw_data, report)
        else:
            populate_stat_report_essentials(coin, data, raw_data, report)

        n_votes = [0, 0, 0] # buy x, hodl, sell y
        n_weights = [0, 0, 0]
        best_model_signal = 3 # set out of bounds to begin with

        # get the best performing models
        models = get_models(best_models)

        for i in range(len(models)):
            # set to prediction mode
            model = models[i]
            model.eval()
            # make the data pytorch compatible
            feature_tensor = torch.tensor([data], dtype=torch.float32)

            with torch.no_grad():
                output = model(feature_tensor)

            if i == 0:
                best_model_signal = int(torch.argmax(output, dim=1))

            for i in range(len(n_votes)):
                n_weights[i] += float(output[0][i])
                n_votes[int(torch.argmax(output, dim=1))] += 1

        # tabulate answers according to different metrics
        n_votes = torch.tensor(n_votes, dtype=torch.float32)
        n_weights = torch.tensor(n_weights, dtype=torch.float32)
        signal_v = DECISIONS[torch.argmax(n_votes)]
        signal_w = DECISIONS[torch.argmax(n_weights)]
        signal_b = DECISIONS[best_model_signal]

        formatted_w_list = [round((x/len(best_models)), 4) for x in n_weights.tolist()]

        buy_signal = formatted_w_list[0]
        hodl_signal = formatted_w_list[1]
        sell_signal = formatted_w_list[2]

        # calculate signal strength
        signal_strength, buy_or_sell = get_signal_strength(data, raw_data)

        report.append("\nAction Signals")
        report.append(f"Signal by best nn:\t{signal_b}")
        report.append(f"Signal by votes:\t{signal_v}")
        report.append(f"Signal by weights:\t{signal_w}")
        report.append(f"Buy/Sell pressure:\t{buy_or_sell} {signal_strength:.1f}X")

        report.append("\nWeight Breakdown")
        report.append("[Greater disparities mean a more confident signal]")
        report.append(f"Weights:\t{formatted_w_list}")
        report.append(f"Diff BUY and SELL:\t\t{abs(buy_signal - sell_signal):>9.4f}")
        report.append(f"Diff HODL and BUY:\t\t{abs(hodl_signal - buy_signal):>9.4f}")
        report.append(f"Diff HODL and SELL:\t\t{abs(hodl_signal - sell_signal):>9.4f}")
        report.append("[>0.7 is strong; <0.3 is weak]")
        report.append(f"BUY to SELL diff ratio:\t{abs(buy_signal - sell_signal) / (abs(hodl_signal - buy_signal) + abs(hodl_signal - sell_signal)):>9.4f}")

    return report



def generate_report(report: List[str], full_report: bool = False) -> None:
    if full_report:
        with open(f"reports/daily/Daily_Report_{str(date.today())}_full.txt", "w") as f:
            # starting from index 1 to avoid first triple space divider
            for row in report[1:]:
                f.write(row + "\n")
    else:
        with open(f"reports/daily/Daily_Report_{str(date.today())}.txt", "w") as f:
            # starting from index 1 to avoid first triple space divider
            for row in report[1:]:
                f.write(row + "\n")



def main() -> None:
    # collects new data and then cleans it
    fetch_data = input("Fetch most recent daily data? [y/n; only if you haven't already fetched today] ")
    if (fetch_data.lower())[0] == 'y':
        days_back = -1
        while days_back < 0:
            days_back = int(input("How many days worth of data? [e.g., 5 if you haven't calculated a signal for 5 days] "))
        fetch_new_data(days_back)
        process_new_data()

    # determines signal and creates report
    full_report = input("Full report? [y/n; y gives all the gory details] ")
    if (full_report.lower())[0] == 'y':
        report = generate_signals(True)
        generate_report(report, True)
    else:
        report = generate_signals()
        generate_report(report)




if __name__ == "__main__":
	main()

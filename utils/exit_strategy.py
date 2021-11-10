from typing import List, Tuple

USD_NTD = 27


def determine_profit_margin(exit_pct: List[int], coin_data: Tuple[str, int, int, int]) -> int:
    #  print(f"{coin_data[0]} exit for {exit_pct}")

    sell_amt = [(x * coin_data[2] / 100) for x in exit_pct]

    total = 0
    price = coin_data[1]
    delta = coin_data[3]

    for i, amt in enumerate(sell_amt):
        sale = USD_NTD * (amt * price) / 10000
        #  print(f"Sale #{i} value = {sale} @ {amt/100} coins")
        price += delta
        total += sale

    return total



def run():
    exit_pcts = [[25, 35, 40],
                 [20, 30, 50],
                 [15, 25, 60],
                 [25, 25, 25, 25],
                 [15, 20, 30, 35],
                 [20, 20, 20, 20, 20],
                 [15, 20, 20, 25, 30]]

    coins_data = [("ADA", 600, 27000, 50, 27000),
                  ("ALGO", 275, 26600, 50, 26600),
                  ("MATIC", 350, 36700, 25, 0),
                  ("SOL", 55000, 792, 5000, 1000)]

    for exit_pct in exit_pcts:
        total_profit = 0
        total_value = 0
        for coin_data in coins_data:
            total_profit += determine_profit_margin(exit_pct, coin_data)
            value = USD_NTD * (coin_data[1] + (coin_data[3]*len(exit_pct))) * coin_data[4] / 10000
            total_value += value
            print(f"Total remaining {coin_data[0]} value = {value}")
        print(f"Total profit for {exit_pct} = {total_profit}")
        total_value += USD_NTD * 100000 * 0.16
        total_value += USD_NTD * 10000 * 1
        print(f"Total remaining = {total_value}")



if __name__ == "__main__":
    run()

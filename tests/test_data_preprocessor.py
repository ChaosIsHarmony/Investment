'''
RUN $ python3 -m tests.test_data_preprocessor
'''
import utils.model_generation_engine.data_preprocessor as dpp

import pandas as pd



def test_handle_missing_data():
        coin = "fakecoin"
        start_date = "2020-10-01"
        end_date = "2020-10-04"

        # missing dates
        data = [['2020-10-01', 1, 1, 1],
                        #comment out the following line to make test fail
                        ['2020-10-02', 0, 0, 0],
                        ['2020-10-03', 0, 0, 0],
                        ['2020-10-04', 3, 3, 3]]
        data = pd.DataFrame(data, columns=["date", 1, 2, 3])
        data = dpp.handle_missing_data(coin, data, start_date, end_date)

        # date type
        assert type(data.iloc[0, 0]) == pd._libs.tslibs.timestamps.Timestamp and type(data.iloc[1, 0]) == pd._libs.tslibs.timestamps.Timestamp and type(data.iloc[2, 0]) == pd._libs.tslibs.timestamps.Timestamp and type(data.iloc[3, 0]) == pd._libs.tslibs.timestamps.Timestamp, "Date column type changed test failed"


        # zero in the middle
        data = [['2020-10-01', 1, 1, 1],
                        ['2020-10-02', 0, 0, 0],
                        ['2020-10-03', 3, 3, 3],
                        ['2020-10-04', 3, 3, 3]]
        data = pd.DataFrame(data, columns=["date", 1, 2, 3])
        data = dpp.handle_missing_data(coin, data, start_date, end_date)

        assert data.iloc[0, 1] == 1.0, "Zero in the middle (first row, second column) test failed."
        assert data.iloc[0, 2] == 1.0, "Zero in the middle (first row, third column) test failed."
        assert data.iloc[0, 3] == 1.0, "Zero in the middle (first row, final column) test failed."

        assert data.iloc[1, 1] == 2.0, "Zero in the middle (second row, second column) test failed."
        assert data.iloc[1, 2] == 2.0, "Zero in the middle (second row, third column) test failed."
        assert data.iloc[1, 3] == 2.0, "Zero in the middle (second row, final column) test failed."

        assert data.iloc[2, 1] == 3.0, "Zero in the middle (third row, second column) test failed."
        assert data.iloc[2, 2] == 3.0, "Zero in the middle (third row, third column) test failed."
        assert data.iloc[2, 3] == 3.0, "Zero in the middle (third row, final column) test failed."

        assert data.iloc[3, 1] == 3.0, "Zero in the middle (final row, second column) test failed."
        assert data.iloc[3, 2] == 3.0, "Zero in the middle (final row, third column) test failed."
        assert data.iloc[3, 3] == 3.0, "Zero in the middle (final row, final column) test failed."


        # consecutive zeros
        data = [['2020-10-01', 1, 1, 1],
                        ['2020-10-02', 0, 0, 0],
                        ['2020-10-03', 0, 0, 0],
                        ['2020-10-04', 3, 3, 3]]
        data = pd.DataFrame(data, columns=["date", 1, 2, 3])
        data = dpp.handle_missing_data(coin, data, start_date, end_date)

        assert data.iloc[0, 1] == 1.0, "Consecutive zeros (first row, second column) test case failed."
        assert data.iloc[0, 2] == 1.0, "Consecutive zeros (first row, third column) test case failed."
        assert data.iloc[0, 3] == 1.0, "Consecutive zeros (first row, final column) test case failed."

        assert data.iloc[1, 1] == 2.0, "Consecutive zeros (second row, second column) test case failed."
        assert data.iloc[1, 2] == 2.0, "Consecutive zeros (second row, third column) test case failed."
        assert data.iloc[1, 3] == 2.0, "Consecutive zeros (second row, final column) test case failed."

        assert data.iloc[2, 1] == 2.5, "Consecutive zeros (third row, second column) test case failed."
        assert data.iloc[2, 2] == 2.5, "Consecutive zeros (third row, third column) test case failed."
        assert data.iloc[2, 3] == 2.5, "Consecutive zeros (third row, final column) test case failed."

        assert data.iloc[3, 1] == 3.0, "Consecutive zeros (final row, second column) test case failed."
        assert data.iloc[3, 2] == 3.0, "Consecutive zeros (final row, third column) test case failed."
        assert data.iloc[3, 3] == 3.0, "Consecutive zeros (final row, final column) test case failed."


        # leading zero
        data = [['2020-10-01', 0, 0, 0],
                        ['2020-10-02', 1, 1, 1],
                        ['2020-10-03', 0, 0, 0],
                        ['2020-10-04', 3, 3, 3]]
        data = pd.DataFrame(data, columns=["date", 1, 2, 3])
        data = dpp.handle_missing_data(coin, data, start_date, end_date)

        assert data.iloc[0, 1] == 1.0, "Leading zero (first row, second column) test failed."
        assert data.iloc[0, 2] == 1.0, "Leading zero (first row, third column) test failed."
        assert data.iloc[0, 3] == 1.0, "Leading zero (first row, final column) test failed."

        assert data.iloc[1, 1] == 1.0, "Leading zero (second row, second column) test failed."
        assert data.iloc[1, 2] == 1.0, "Leading zero (second row, third column) test failed."
        assert data.iloc[1, 3] == 1.0, "Leading zero (second row, final column) test failed."

        assert data.iloc[2, 1] == 2.0, "Leading zero (third row, second column) test failed."
        assert data.iloc[2, 2] == 2.0, "Leading zero (third row, third column) test failed."
        assert data.iloc[2, 3] == 2.0, "Leading zero (third row, final column) test failed."

        assert data.iloc[3, 1] == 3.0, "Leading zero (final row, second column) test failed."
        assert data.iloc[3, 2] == 3.0, "Leading zero (final row, third column) test failed."
        assert data.iloc[3, 3] == 3.0, "Leading zero (final row, final column) test failed."


        # ending zero
        data = [['2020-10-01', 1, 1, 1],
                        ['2020-10-02', 1, 1, 1],
                        ['2020-10-03', 2, 2, 2],
                        ['2020-10-04', 0, 0, 0]]
        data = pd.DataFrame(data, columns=["date", 1, 2, 3])
        data = dpp.handle_missing_data(coin, data, start_date, end_date)

        assert data.iloc[0, 1] == 1.0, "Ending zero (first row, second column) test failed."
        assert data.iloc[0, 2] == 1.0, "Ending zero (first row, third column) test failed."
        assert data.iloc[0, 3] == 1.0, "Ending zero (first row, final column) test failed."

        assert data.iloc[1, 1] == 1.0, "Ending zero (second row, second column) test failed."
        assert data.iloc[1, 2] == 1.0, "Ending zero (second row, third column) test failed."
        assert data.iloc[1, 3] == 1.0, "Ending zero (second row, final column) test failed."

        assert data.iloc[2, 1] == 2.0, "Ending zero (third row, second column) test failed."
        assert data.iloc[2, 2] == 2.0, "Ending zero (third row, third column) test failed."
        assert data.iloc[2, 3] == 2.0, "Ending zero (third row, final column) test failed."

        assert data.iloc[3, 1] == 2.0, "Ending zero (final row, second column) test failed."
        assert data.iloc[3, 2] == 2.0, "Ending zero (final row, third column) test failed."
        assert data.iloc[3, 3] == 2.0, "Ending zero (final row, final column) test failed."



def test_normalize_data():
        # check fear greed are out of 100
        data = [['2020-10-01', 1, 1, 1],
                        ['2020-10-02', 1, 1, 1],
                        ['2020-10-03', 4, 4, 4],
                        ['2020-10-04', 2, 2, 2]]
        data = pd.DataFrame(data, columns=["date", "fear_greed", 2, 3])
        data = dpp.normalize_data(data)

        assert data.iloc[0,1] == 1/100, "Normalize data (first row) fear_greed index test failed."
        assert data.iloc[1,1] == 1/100, "Normalize data (second row) fear_greed index test failed."
        assert data.iloc[2,1] == 4/100, "Normalize data (third row) fear_greed index test failed."
        assert data.iloc[3,1] == 2/100, "Normalize data (final row) fear_greed index test failed."


        # standard
        data = [['2020-10-01', 1, 1, 1],
                        ['2020-10-02', 1, 1, 1],
                        ['2020-10-03', 4, 4, 4],
                        ['2020-10-04', 2, 2, 2]]
        data = pd.DataFrame(data, columns=["date", "fear_greed", 2, 3])
        data = dpp.normalize_data(data)

        assert data.iloc[0,2] == 1, "Normalize data (first row) standard test failed."
        assert data.iloc[1,2] == 1, "Normalize data (second row) standard test failed."
        assert data.iloc[2,2] == 1, "Normalize data (third row) standard test failed."
        assert data.iloc[3,2] == 1/3, "Normalize data (final row) standard test failed."


        # with zeros
        data = [['2020-10-01', 0, 0, 0],
                        ['2020-10-02', 1, 1, 1],
                        ['2020-10-03', 4, 4, 4],
                        ['2020-10-04', 2, 2, 2]]
        data = pd.DataFrame(data, columns=["date", "fear_greed", 2, 3])
        data = dpp.normalize_data(data)

        assert data.iloc[0,2] == 1, "Normalize data (first row) w/ zeros test failed."
        assert data.iloc[1,2] == 1, "Normalize data (second row) w/ zeros test failed."
        assert data.iloc[2,2] == 1, "Normalize data (third row) w/ zeros test failed."
        assert data.iloc[3,2] == 1/2, "Normalize data (final row) w/ zeros test failed."


        # with only zeros
        data = [['2020-10-01', 0, 0, 0],
                        ['2020-10-02', 0, 0, 0],
                        ['2020-10-03', 0, 0, 0],
                        ['2020-10-04', 0, 0, 0]]
        data = pd.DataFrame(data, columns=["date", "fear_greed", 2, 3])
        data = dpp.normalize_data(data)
        assert data.iloc[0,2] == 1, "Normalize data (first row) w/ only zeros test failed."
        assert data.iloc[1,2] == 0, "Normalize data (second row) w/ only zeros test failed."
        assert data.iloc[2,2] == 0, "Normalize data (third row) w/ only zeros test failed."
        assert data.iloc[3,2] == 0, "Normalize data (final row) w/ only zeros test failed."



def test_calculate_price_SMAs():
        data = [['x', 0]]
        for i in range(1,500):
                data.append(['x', data[i-1][1]+1])
        data = pd.DataFrame(data, columns=["date", "price"])
        # reverses dataframe as it does in the calculate_SMAs() method in the data_preprocessor.py file that calls the method being tested
        data = data.reindex(index=data.index[::-1]).reset_index()
        data = data.drop(columns=["index"])

        data = dpp.calculate_price_SMAs(data)
        '''
        (498+497+496+495+494)/5 == data.iloc[5,2]
        Must start at 6th item from the last in order to be able to subtract the first item added to the moving total
        '''
        total = lambda x,y: sum([n for n in range(x, y+1)])
        assert total(494,498)/5 == data.iloc[5,2], "The 5-day price SMA test failed."
        assert total(489,498)/10 == data.iloc[10,3], "The 10-day price SMA test failed."
        assert total(474,498)/25 == data.iloc[25,4], "The 25-day price SMA test failed."
        assert total(449,498)/50 == data.iloc[50,5], "The 50-day price SMA test failed."
        assert total(424,498)/75 == data.iloc[75,6], "The 75-day price SMA test failed."
        assert total(399,498)/100 == data.iloc[100,7], "The 100-day price SMA test failed."
        assert total(349,498)/150 == data.iloc[150,8], "The 150-day price SMA test failed."
        assert total(299,498)/200 == data.iloc[200,9], "The 200-day price SMA test failed."
        assert total(249,498)/250 == data.iloc[250,10], "The 250-day price SMA test failed."
        assert total(199,498)/300 == data.iloc[300,11], "The 300-day price SMA test failed."
        assert total(149,498)/350 == data.iloc[350,12], "The 350-day price SMA test failed."



def test_calculate_fear_greed_SMAs():
        data = [['x', 0]]
        for i in range(1,500):
                data.append(['x', data[i-1][1]+1])
        data = pd.DataFrame(data, columns=["date", "fear_greed"])
        # reverses dataframe as it does in the calculate_SMAs() method in the data_preprocessor.py file that calls the method being tested
        data = data.reindex(index=data.index[::-1]).reset_index()
        data = data.drop(columns=["index"])

        data = dpp.calculate_fear_greed_SMAs(data)
        '''
        (498+497+496)/3 == data.iloc[3,2]
        Must start at 4th item from the last in order to be able to subtract the first item added to the moving total
        '''
        total = lambda x,y: sum([n for n in range(x, y+1)])
        assert total(496,498)/3 == data.iloc[3,2], "The 3-day fear/greed SMA test failed."
        assert total(494,498)/5 == data.iloc[5,3], "The 5-day fear/greed SMA test failed."
        assert total(492,498)/7 == data.iloc[7,4], "The 7-day fear/greed SMA test failed."
        assert total(490,498)/9 == data.iloc[9,5], "The 9-day fear/greed SMA test failed."
        assert total(488,498)/11 == data.iloc[11,6], "The 11-day fear/greed SMA test failed."
        assert total(486,498)/13 == data.iloc[13,7], "The 13-day fear/greed SMA test failed."
        assert total(484,498)/15 == data.iloc[15,8], "The 15-day fear/greed SMA test failed."
        assert total(469,498)/30 == data.iloc[30,9], "The 30-day fear/greed SMA test failed."



def test_get_signal_value():
    signalDefault = dpp.get_signal_value(0)
    signalSell = dpp.get_signal_value(-0.176)
    signalBuy = dpp.get_signal_value(0.151)

    assert signalDefault == 1, "Failed the get_signal_value test for the default hodl value."
    assert signalSell == 2, "Failed the get_signal_value test for the sell value."
    assert signalBuy == 0, "Failed the get_signal_value test for the buy value."



def test_get_weighting_constant():
        w_const1 = dpp.get_weighting_constant(4)
        w_const2 = dpp.get_weighting_constant(7)

        assert 0.099 < w_const1 < 0.11, f"WRONG VALUE: 0.099 < {w_const1} < 0.11 | Failed get_weighting_constant test."
        assert 0.0356 < w_const2 < 0.0358, f"WRONG VALUE: 0.0356 < {w_const2} 0.0358 | Failed get_weighting_constant test."



def test_calculate_signals():
        '''
        Based on the specific model where signals change at different threshholds (see get_signal() in data_preprocessor.py). Change assertions if model changes.
        '''
        # ascending
        data = []
        for i in range(1, 100):
                data.append([i])

        data = pd.DataFrame(data, columns=["price"])
        data = dpp.calculate_signals(data, 7)

        # 7 -> 14 : ((14-7) / 7) * weight_const * 7 = 0.25
        # 7 -> 13 : ((13-7) / 7) * weight_const * 6 = 0.184
        # 7 -> 12 : I(12-7) / 7) * weight_const * 5 = 0.128
        # 7 -> 11 : ((11-7) / 7) * weight_const * 4 = 0.082
        # 7 -> 10 : ((10-7) / 7) * weight_const * 3 = 0.046
        # 7 -> 9 : ((9-7) / 7) * weight_const * 2 = 0.020
        # 7 -> 8 : ((8-7) / 7) * weight_const * 1 = 0.005
        # avg = 0.714
        assert data["signal"][6] == 0, f"WRONG VALUE: {data['signal'][6]:.4f} != 0 | Failed BUY 3X signal in test_calculate_signals test."
        assert data["signal"][49] == 1, f"WRONG VALUE: {data['signal'][33]:.4f} != 3 | Failed positive HODL signal in test_calculate_signals test."

        # descending
        data = []
        for i in range(1, 100):
                data.append([1/i])

        data = pd.DataFrame(data, columns=["price"])
        data = dpp.calculate_signals(data, 7)

        assert data["signal"][11] == 2, f"WRONG VALUE: {data['signal'][11]:.4f} != 4 | Failed SELL Y signal in test_calculate_signals test."
        assert data["signal"][45] == 1, f"WRONG VALUE: {data['signal'][27]:.4f} != 3 | Failed HODL signal in test_calculate_signals test."



def run_data_preprocessor_tests():
        test_handle_missing_data()
        print("test_handle_missing_data() tests all passed.")
        test_normalize_data()
        print("test_normalize_data() tests all passed.")
        test_calculate_price_SMAs()
        print("test_calculate_price_SMAs() tests all passed.")
        test_calculate_fear_greed_SMAs()
        print("test_calculate_fear_greed_SMAs() tests all passed.")
        test_get_signal_value()
        print("test_get_signal_value() tests all passed.")
        test_get_weighting_constant()
        print("test_get_weighting_constant() tests all passed.")
        test_calculate_signals()
        print("test_calculate_signals() tests all passed.")



if __name__ == "__main__":
    run_data_preprocessor_tests()


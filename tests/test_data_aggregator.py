'''
RUN $ python3 -m tests.test_data_aggregator
'''
import utils.model_generation_engine.data_aggregator as da

import time
from datetime import date, datetime



def test_get_time():
    total = 0
    for _ in range(1000):
        start = da.get_time()
        time.sleep(0.001)
        stop = da.get_time()
        total += stop-start

    assert 1500 > (total) > 1000, "Failed get_time test."



def test_get_correct_date_format():
    wonky_date_str = "2021-06-14"
    wonky_date = datetime.strptime(wonky_date_str, '%Y-%m-%d')
    correct_date = da.get_correct_date_format(wonky_date)

    assert str(correct_date) == "14-06-2021", "Failed get_correct_date_format test."



def test_extract_basic_data():
    # valid data
    date = "18-01-2021"
    data = {
            "market_data": { "current_price": { "twd": 1 }, "market_cap": { "twd": 2 }, "total_volume": { "twd": 3 } },
            "community_data": { "fb": 4, "reddit": 5 },
            "dev_stats": { },
            "public_interest_stats": { "alexa": 6, "bing": 7 }
    }
    data_dict = da.extract_basic_data(data, date)

    assert data_dict["date"] == date, "Failed extract date in extract_basic_data test."
    assert data_dict["price"] == 1, "Failed extract price in extract_basic_data test."
    assert data_dict["market_cap"] == 2, "Failed extract market cap in extract_basic_data test."
    assert data_dict["volume"] == 3, "Failed extract volume in extract_basic_data test."

    # bad data
    data = { "this...": "...contains none of the information needed to be extracted and should all result in zero values." }
    data_dict = da.extract_basic_data(data, date)

    assert data_dict["date"] == date, "Failed extract date in bad data extract_basic_data test."
    assert data_dict["price"] == 0, "Failed extract price in bad data extract_basic_data test."
    assert data_dict["market_cap"] == 0, "Failed extract market cap in bad data extract_basic_data test."
    assert data_dict["volume"] == 0, "Failed extract volume in bad data extract_basic_data test."



def run_data_aggregator_tests():
    test_get_time()
    print("test_get_time() tests all passed.")
    test_get_correct_date_format()
    print("test_get_correct_date_format() tests all passed.")
    test_extract_basic_data()
    print("test_extract_basic_data() tests all passed.")



if __name__ == "__main__":
    run_data_aggregator_tests()

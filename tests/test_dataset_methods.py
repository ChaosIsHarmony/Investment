'''
RUN $ python3 -m tests.test_dataset_methods
'''
import utils.model_generation_engine.data_preprocessor as dpp
import utils.model_generation_engine.dataset_methods as dm
import utils.model_generation_engine.neural_nets as nn

import os
import pandas as pd



def test_generate_dataset():
    # create a communal dataset to use for all tests in this section
    data = []
    for _ in range(7):
            row = [x+1 for x in range(nn.N_FEATURES)]
            data.append(row)
    data[0].append(0)
    data[1].append(1)
    data[2].append(2)
    data[3].append(2)
    data[4].append(3)
    data[5].append(3)
    data[6].append(3)
    data = pd.DataFrame(data, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, "signal"])

    # No augmentation
    altered_data = dm.generate_dataset(data, len(data), 0)
    assert len(altered_data) == 7, "No augmentation dataset length test failed"
    assert len(altered_data[0]) == 2, "No augmentation feature/target tuple length test failed"
    assert len(altered_data[0][0]) == nn.N_FEATURES, "No augmentation feature vector length test failed."

    # 10x augmentation
    # 10*3 for signal 0
    # 10*3 for signal 1
    # 10*2 for signal 2
    # 10*2 for signal 2
    # 10*1 for signal 3
    # 10*1 for signal 3
    # 10*1 for signal 3
    # + 7 for the original elements in the data
    altered_data = dm.generate_dataset(data, len(data), 0, 10)
    assert len(altered_data) == (10*3)+(10*3)+(10*2)+(10*2)+(10*1)+(10*1)+(10*1)+7, "10x augmentation dataset length test failed"
    assert len(altered_data[0]) == 2, "10x augmentation feature/target tuple length test failed"
    assert len(altered_data[0][0]) == nn.N_FEATURES, "10x augmentation feature vector length test failed."

    # Testing offset
    altered_data = dm.generate_dataset(data, len(data), 2, 10)
    assert len(altered_data) == (10*2)+(10*2)+(10*1)+(10*1)+(10*1)+5, "2 offset dataset length test failed"
    assert len(altered_data[0]) == 2, "2 offset feature/target tuple length test failed"
    assert len(altered_data[0][0]) == nn.N_FEATURES, "2 offset feature vector length test failed."

    # Testing limit
    altered_data = dm.generate_dataset(data, len(data)-2, 0, 10)
    assert len(altered_data) == (10*2)+(10*2)+(10*1)+(10*1)+(10*2)+5, "2 limit dataset length test failed"
    assert len(altered_data[0]) == 2, "2 limit feature/target tuple length test failed"
    assert len(altered_data[0][0]) == nn.N_FEATURES, "2 limit feature vector length test failed."



def create_fake_csv():
    data = []
    for i in range(100):
            dt_list = []
            # +2 is for date and signal
            for j in range(nn.N_FEATURES+2):
                    dt_list.append(i)
            data.append(dt_list)

    col_labels = ["date"]
    for i in range(nn.N_FEATURES):
            col_labels.append(i)
    col_labels.append("signal")

    data = pd.DataFrame(data, columns=col_labels)
    data = dpp.normalize_data(data)
    data.to_csv("datasets/complete/fakecoin_historical_data_complete.csv", index=False)

    return data



def destroy_fake_coin():
    os.remove("datasets/complete/fakecoin_historical_data_complete.csv")



def test_get_datasets():
    coin = "fakecoin"
    data = create_fake_csv()
    train_data, valid_data, test_data = dm.get_datasets(coin, data_aug_factor=16)

    # test with standard 16x augmentation
    # len(data)*0.7*16 = the data augmentation portion
    # + (len(data)*0.7) = the original datapoints before augmentation
    assert len(train_data) == (len(data)*0.7*16) + (len(data)*0.7), "Failed train_data size test in get_datasets test."
    assert len(valid_data) == len(data)*0.15, "Failed valid_data size test in get_datasets test."
    assert len(test_data) == len(data)*0.15, "Failed test_data size test in get_datasets test."
    assert 0.999 < train_data[int(len(data)*0.7*16)+69][0][0] < 1.001, "Failed train_data value test in get_datasets test."
    assert valid_data[int(len(data)*0.15)-1][0][0] == 1.0, "Failed valid_data value test in get_datasets test."
    assert test_data[int(len(data)*0.15)-1][0][0] == 1.0, "Failed test_data value test in get_datasets test."

    # test with 0 augmentation
    train_data, valid_data, test_data = dm.get_datasets(coin)

    assert len(train_data) == len(data)*0.7, "Failed train_data size test in get_datasets test."
    assert len(valid_data) == len(data)*0.15, "Failed valid_data size test in get_datasets test."
    assert len(test_data) == len(data)*0.15, "Failed test_data size test in get_datasets test."
    assert train_data[int(len(data)*0.7)-1][0][0] == 1.0, "Failed train_data value test in get_datasets test."
    assert valid_data[int(len(data)*0.15)-1][0][0] == 1.0, "Failed valid_data value test in get_datasets test."
    assert test_data[int(len(data)*0.15)-1][0][0] == 1.0, "Failed test_data value test in get_datasets test."

    destroy_fake_coin()



def test_shuffle_data():
    data = [[0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9]]

    data = dm.shuffle_data(data)

    assert (data[0] == [0] and data[1] == [1] and data[2] == [2] and data[3] == [3] and data[4] == [4] and data[5] == [5] and data[6] == [6] and data[7] == [7] and data[8] == [8] and data[9] == [9]) == False, "Failed random shuffling of data in shuffle_data test."



def run_dataset_methods_tests():
    test_generate_dataset()
    print("test_generate_dataset() tests all passed.")
    test_get_datasets()
    print("test_get_datasets() tests all passed.")
    test_shuffle_data()
    print("test_shuffle_data() tests all passed.")



if __name__ == "__main__":
    run_dataset_methods_tests()

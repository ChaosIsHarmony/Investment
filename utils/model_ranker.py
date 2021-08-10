import os
import glob
import pandas as pd
from . import common
import time
from typing import List, Tuple


def prepare_datasets(coin: str) -> Tuple[List[Tuple[List[float], float]], List[Tuple[List[float], float]], List[Tuple[List[float], float]]]:
    start_time = time.time()
    data_aug_factor = 0
    print("Creating datasets...")

    # separate valid, test
    try:
        train_data, valid_data, test_data = common.get_datasets(coin, data_aug_factor)
    except:
        raise

    # all data together
    data = train_data + valid_data + test_data

    print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")

    return data, valid_data, test_data




def prune_models_by_accuracy(coin: str):
    # load data
    data, valid_data, test_data = prepare_datasets(coin)

    # test all models to find accuracy stats
    scores = []
    filenames = glob.glob(f"models/aggregate/{coin}*")

    least_reliable_models = []
    for filename in filenames:
        params = common.get_model_params(coin, filename)
        if params == None:
            print(f"{filename} Not Found. Continuing on to other models.")
            continue

        model = common.load_model_by_params(filename, params)

        # evaluate
        model = common.load_model(model, f"{filename}")
        model_acc_all = common.evaluate_model(model, data)
        model_acc_valid = common.evaluate_model(model, valid_data)
        model_acc_test = common.evaluate_model(model, test_data)

        if (model_acc_all[0] > 0.55) and (model_acc_valid[0] > 0.45) and (model_acc_test[0] > 0.7):
            print(f"{filename}")
            print("ALL DATA")
            common.print_evaluation_status(model_acc_all)
            print("VALIDATION DATA")
            common.print_evaluation_status(model_acc_valid)
            print("TEST DATA")
            common.print_evaluation_status(model_acc_test)
        else:
            least_reliable_models.append(filename)
            print(f"Model {filename} performance did not meet the threshold.")

    # prune the weakest models
    for f in least_reliable_models:
        try:
            os.remove(f)
        except:
            print(f"Error when attempting to remove {f}.")





def benchmark_models(model_coin: str, test_coin: str) -> None:
    '''
    Cross compares model performance on other datasets, e.g., bitcoin models on the ethereum dataset.
    '''
    # load data
    data, valid_data, test_data = prepare_datasets(test_coin)

    # create list of best models
    best_models = []
    with open(f"reports/{model_coin}_best_performers_all.txt", 'r') as f:
        best_models = f.read().splitlines()

    most_reliable_models = []
    for filename in best_models:
        model_params = common.get_model_params(model_coin, filename)
        if model_params == None:
            print(f"{filename} Not Found. Continuing on to other models.")
            continue

        model = common.load_model_by_params(filename, model_params)

        # evaluate
        model = common.load_model(model, f"{filename}")
        model_acc_all = common.evaluate_model(model, data)
        model_acc_valid = common.evaluate_model(model, valid_data)
        model_acc_test = common.evaluate_model(model, test_data)

        if (model_acc_all[0] > 0.5) and (model_acc_valid[0] > 0.1) and (model_acc_test[0] > 0.4):
            print(f"{filename}")
            print("ALL DATA")
            common.print_evaluation_status(model_acc_all)
            print("VALIDATION DATA")
            common.print_evaluation_status(model_acc_valid)
            print("TEST DATA")
            common.print_evaluation_status(model_acc_test)
            most_reliable_models.append(filename)
        else:
            print(f"Model {filename} performance did not meet the threshold.")

    print(most_reliable_models)



if __name__ == "__main__":
    #benchmark_models(model_coin="all", test_coin="ethereum")
    prune_models_by_accuracy("all")

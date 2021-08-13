import glob
import os
import torch
from . import neural_nets as nn
from .. import common
from typing import List, Tuple


#
# ---------- PARSE PARAMS FROM FILE ----------
#
def extract_datum(reports: str, key: str, terminator: str) -> str:
    start_ind = reports.find(key) + len(key)
    end_ind = reports.find(terminator, start_ind)

    return reports[start_ind : end_ind].strip()



def parse_reports(coin: str, model_architecture: str) -> List[dict]:
    try:
        with open(f"reports/{coin}_Parameter_Tuning_Report_{model_architecture}.txt", 'r') as f:
            reports = f.read()
    except:
        print(f"reports/{coin}_Parameter_Tuning_Report_{model_architecture}.txt not found.")
        raise

    models = []
    while len(reports) > 10:
        model = {}
        model["model_num"] =  int(extract_datum(reports, "MODEL:", '\n'))
        model["architecture"] = int(extract_datum(reports, "PARAMETERS:\n\tHidden_", 'e'))
        #  model["architecture"] = extract_datum(reports, "PARAMETERS:", 'e')
        model["eta"] = float(extract_datum(reports, "eta:", '|'))
        model["eta_decay"] = float(extract_datum(reports, "decay:", '|'))
        model["dropout"] = float(extract_datum(reports, "dropout:", '\n'))
        model["accuracy"] = float(extract_datum(reports, "Decision:", '\n'))
        model["inaccuracy"] = float(extract_datum(reports, "Opposite:", '\n'))

        # Add model if meets accuracy threshhold
        if model["accuracy"] > common.PROMISING_ACCURACY_THRESHOLD and model["inaccuracy"] < common.INACCURACY_THRESHOLD:
            models.append(model)

        # delete up until next model
        start_ind = reports.find("MODEL", reports.find("Opposite"))
        reports = reports[start_ind:]

    return models



def list_promising_model_details(coin: str, model_architecture: str) -> None:
    models = parse_reports(coin, model_architecture)

    count = 0
    for model in models:
        count += 1
        print(f"Model num: {model['model_num']}")
        print(f"Model acc: {model['accuracy']}")
        print(f"Model bad: {model['inaccuracy']}")
        print()

        print(f"{count} promising models found.")



def get_model_params(coin: str, filename: str) -> dict:
    start_ind = filename.find('_') + 1
    end_ind = filename.find('_', start_ind) + 2
    model_architecture = filename[start_ind:end_ind]

    start_ind = filename.find('_', end_ind) + 1
    end_ind = filename.find('_', start_ind)
    model_num = filename[start_ind:end_ind]

    try:
        models = parse_reports(coin, model_architecture)
        for model in models:
            if model["model_num"] == int(model_num):
                return model
    except:
        return None



#
# ---------- SAVE/LOAD MODELS ----------
#
def save_model(model: nn.CryptoSoothsayer, filepath: str) -> None:
    torch.save(model.state_dict(), filepath)



def load_model(model_to_load: nn.CryptoSoothsayer, filepath: str) -> nn.CryptoSoothsayer:
    model = model_to_load
    model.load_state_dict(torch.load(filepath))

    return model



def load_model_by_params(filepath: str, params: dict) -> nn.CryptoSoothsayer:
    return nn.create_model(hidden_layer_size = params["architecture"], dropout = params["dropout"], eta = params["eta"], eta_decay = params["eta_decay"])


# Legacy method; will delete well all models are trained using new specification
def load_pretrained_model(filepath: str) -> nn.CryptoSoothsayer:
    start_ind = filepath.find('_') + 1
    end_ind = filepath.find('_', start_ind) + 2
    model_architecture = filepath[start_ind:end_ind]

    if "Laptop_0" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 20, nn.N_SIGNALS, 0, 1, 1)
    elif "Laptop_1" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 21, nn.N_SIGNALS, 0, 1, 1)
    elif "Laptop_2" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 22, nn.N_SIGNALS, 0, 1, 1)
    elif "Laptop_3" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 23, nn.N_SIGNALS, 0, 1, 1)
    elif "Pi_0" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 5, nn.N_SIGNALS, 0, 1, 1)
    elif "Pi_1" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 6, nn.N_SIGNALS, 0, 1, 1)
    elif "Pi_2" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 7, nn.N_SIGNALS, 0, 1, 1)
    elif "Pi_3" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 8, nn.N_SIGNALS, 0, 1, 1)
    elif "Pi_4" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 9, nn.N_SIGNALS, 0, 1, 1)
    elif "Pi_5" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 10, nn.N_SIGNALS, 0, 1, 1)
    elif "Pi_6" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 11, nn.N_SIGNALS, 0, 1, 1)
    elif "Pi_7" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 12, nn.N_SIGNALS, 0, 1, 1)
    elif "PC_0" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 13, nn.N_SIGNALS, 0, 1, 1)
    elif "PC_1" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 14, nn.N_SIGNALS, 0, 1, 1)
    elif "PC_2" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 15, nn.N_SIGNALS, 0, 1, 1)
    elif "PC_3" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 16, nn.N_SIGNALS, 0, 1, 1)
    elif "PC_4" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 17, nn.N_SIGNALS, 0, 1, 1)
    elif "PC_5" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 18, nn.N_SIGNALS, 0, 1, 1)
    elif "PC_6" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 19, nn.N_SIGNALS, 0, 1, 1)
    elif "PC_7" in model_architecture:
        model = nn.CryptoSoothsayer(model_architecture, nn.N_FEATURES, 19, nn.N_SIGNALS, 0, 1, 1)
    else:
        model = None

    return load_model(model, filepath)

#
# ---------- EVALUATE/VALIDATE MODELS ----------
#
def print_evaluation_status(model_accuracy: List[float]) -> str:
    '''
    Prints summary for model evaluation.
    '''
    report = f"""
        POSITIVE:
            [+] Perfect accuracy: {model_accuracy[0]:>10.4f}
        NEGATIVE:
            [-] Told to hodl but should have sold/bought rate: {model_accuracy[1]:>10.4f}
            [--] Should have hodled but told to sell/buy rate: {model_accuracy[2]:>10.4f}
            [---] Told to do the opposite of correct move rate: {model_accuracy[3]:>10.4f}
        """

    print(report)

    return report



def evaluate_model(model: nn.CryptoSoothsayer, test_data: Tuple[List[float], float]) -> List[float]:
    model.eval()
    correct = 0
    safe_fail = 0
    nasty_fail = 0
    catastrophic_fail = 0
    for features, target in test_data:
        feature_tensor, target_tensor = common.convert_to_tensor(model, features, target)

        with torch.no_grad():
            output = model(feature_tensor)

        decision = torch.argmax(output, dim=1)

        # flawless
        if decision == target_tensor:
            correct += 1
            # catastrophic failure (e.g., told to buy when should have sold)
        elif (target_tensor > 1 and decision < 1) or (target_tensor < 1 and decision > 1):
            catastrophic_fail += 1
        # severe failure (e.g., should have hodled but was told to buy or sell
        elif target_tensor == 1 and (decision < 1 or decision > 1):
            nasty_fail += 1
        # decision was to hodl, but should have sold or bought
        else:
            safe_fail += 1

    model_accuracy = [correct/len(test_data), safe_fail/len(test_data), nasty_fail/len(test_data), catastrophic_fail/len(test_data)]


    return model_accuracy



def validate_model(model: nn.CryptoSoothsayer, valid_data: Tuple[List[float], float], lowest_valid_loss: float, filepath: str) -> Tuple[float, float]:
    '''
    Validates the model on the validation dataset.
    Saves model if validation loss is lower than the current lowest.
    Returns the average validation loss and lowest validation loss.
    '''
    # set to evaluate mode to turn off components like dropout
    model.eval()
    valid_loss = 0.0
    for features, target in valid_data:
        # make data pytorch compatible
        feature_tensor, target_tensor = common.convert_to_tensor(model, features, target)
        # model makes prediction
        with torch.no_grad():
            model_output = model(feature_tensor)
            loss = model.get_criterion()(model_output, target_tensor)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss/len(valid_data)

    return avg_valid_loss, lowest_valid_loss



#
# ---------- CULLING METHOD ----------
#
def prune_models_by_accuracy(coin: str) -> None:
    # load data
    data, valid_data, test_data = common.prepare_model_pruning_datasets(coin)

    # test all models to find accuracy stats
    scores = []
    filenames = glob.glob(f"models/aggregate/{coin}*")

    least_reliable_models = []
    most_reliable_model = ["", 0]

    for filename in filenames:
        params = common.get_model_params(coin, filename)
        if params == None:
            print(f"{filename} Not Found. Continuing on to other models.")
            continue

        model = load_model_by_params(filename, params)
        model = load_model(model, f"{filename}")

        # evaluate
        model_acc_all = evaluate_model(model, data)
        model_acc_valid = evaluate_model(model, valid_data)
        model_acc_test = evaluate_model(model, test_data)

        if (model_acc_all[0] > common.PRUNING_THRESHOLD_ALL) and (model_acc_valid[0] > common.PRUNING_THRESHOLD_VALID) and (model_acc_test[0] > common.PRUNING_THRESHOLD_TEST):
            total_acc = model_acc_test[0] + model_acc_valid[0] + model_acc_all[0]
            if total_acc > most_reliable_model[1]:
                most_reliable_model[0] = filename
                most_reliable_model[1] = total_acc
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
    rm_cnt = 0
    for f in least_reliable_models:
        try:
            os.remove(f)
            print(f"Successfully removed {f}.")
            rm_cnt += 1
        except:
            print(f"Error when attempting to remove {f}.")

    print(f"{rm_cnt} weak models removed [{rm_cnt/len(filenames)*100:.2f}% of original models].")
    print(f"Most reliable model: {most_reliable_model[0]}\n\tAvg. Acc.: {100*most_reliable_model[1]/3:.2f}%")

#
# ---------- COMPARISON METHODS ----------
#
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

        model = load_model_by_params(filename, model_params)
        model = load_model(model, f"{filename}")

        # evaluate
        model_acc_all = evaluate_model(model, data)
        model_acc_valid = evaluate_model(model, valid_data)
        model_acc_test = evaluate_model(model, test_data)

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

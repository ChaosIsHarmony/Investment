'''
This file is used to train neural nets.
'''
import os
import glob
import pandas as pd
import torch
from datetime import datetime
import time
import numpy as np
from typing import List, Tuple
from . import neural_nets as nn
from . import common

REPORTS = []

#
# ------------ DELETING FUNCTIONS -----------
#
def cleanup(coin: str, destination_dir: str) -> None:
    '''
    Delete all the *.pt files from models that are not worth keeping.
    '''
    # delete all non-promising models
    file_list = glob.glob("models/*.pt")
    for f in file_list:
        try:
            os.remove(f)
            print(f"Successfully removed {f}")
        except:
            print(f"Error when attempting to remove {f}.")

    # delete trained promising models lowest validation that didn't make the cut
    # NOTE: we are still keeping the promising models to train later with different data aug factors
    file_list = glob.glob("models/promising/*lowest_val_loss.pt")
    for f in file_list:
        try:
            os.remove(f)
            print(f"Successfully removed {f}")
        except:
            print(f"Error when attempting to remove {f}.")

    # move previously promising models to the specified directory
    file_list = glob.glob("models/promising/*.pt")
    for f in file_list:
        try:
            offset = len("models/promising/")
            file_handle = f[offset:]
            os.rename(f, f"models/{destination_dir}/{file_handle}")
            print(f"Successfully moved {f} to {destination_dir}")
        except:
            print(f"Error when attempting to move {f}")

    # move best models to specified directory
    file_list = glob.glob(f"models/best/{coin}_*")
    for f in file_list:
        try:
            offset = len("models/best/")
            file_handle = f[offset:]
            os.rename(f, f"models/{destination_dir}/{file_handle}")
            print(f"Successfully moved {f} to models/{destination_dir}/")
        except:
            print(f"Error when attempting to move {f}")



#
# ---------- REPORTING FUNCTIONS ----------
#
def join_all_reports() -> str:
    '''
    Collects all reports for different steps of the training process into a single string in prep for saving to disk.
    '''
    global REPORTS

    final_report = ""
    for report in REPORTS:
        final_report += report + "\n"

    return final_report



def save_report(coin: str, architecture: str) -> None:
    with open(f"reports/{coin}_{architecture}_report.txt", "w") as report:
        report.write(join_all_reports())



def print_batch_status(avg_train_loss: float, avg_valid_loss:float, start_time: float) -> None:
    '''
    Prints summary for the latest batch to console and appends it to the global REPORTS variable
    '''
    global REPORTS

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    report = f"System Time: {current_time} | Time Elapsed: {(time.time() - start_time) / 60:.1f} mins. | Avg. Training Loss: {avg_train_loss:.4f} | Avg. Validation Loss: {avg_valid_loss:.4f} | eta: {nn.OPTIMIZER.state_dict()['param_groups'][0]['lr']:.6f}"
    REPORTS.append(report)
    print(report)



#
# ----------- TRAINING FUNCTIONS ----------
#
def take_one_step(model: nn.CryptoSoothsayer, features: List[float], target: float) -> float:
    '''
    Forward propogates a single feature vector through the network, the back propogates based on the loss.
    Returns the training loss.
    '''
    # set to train mode here to activate components like dropout
    model.train()
    # make data pytorch compatible
    feature_tensor, target_tensor = common.convert_to_tensor(features, target)
    # Forward
    model_output = model(feature_tensor)
    loss = nn.get_criterion()(model_output, target_tensor)
    # Backward
    nn.get_optimizer().zero_grad()
    loss.backward()
    nn.get_optimizer().step()
    # adjust learning rate
    nn.get_scheduler().step()
    train_loss = loss.item()

    return train_loss



def terminate_early(prev_valid_losses: List[float]) -> bool:
    '''
    Sends signal to terminate early if the validation loss is increasing over a 10-batch interval.
    '''
    if len(prev_valid_losses) >= 10:
        ind = len(prev_valid_losses) - 1
        valid_loss_trend = 0

        while ind > 0:
            valid_loss_trend += prev_valid_losses[ind] - prev_valid_losses[ind-1]
            ind -= 1

        if valid_loss_trend >= 0:
            return True

        # prune oldest item from list
        del prev_valid_losses[0]

    return False



def fully_train(model: nn.CryptoSoothsayer, data: Tuple[List[float], float], start_time: float, filepath: str, n_epochs: int = 20, batch_size: int = 256) -> float:
    global REPORTS

    # unpack training and validation datasets
    train_data, valid_data = data

    epoch = 0
    lowest_valid_loss = np.inf
    last_valid_loss = 0.0
    prev_valid_losses = []

    while epoch < n_epochs:
        # setup
        train_data = common.shuffle_data(train_data)
        steps = 0
        total_train_loss = 0.0
        total_valid_loss = 0.0

        for features, target in train_data:
            steps += 1
            # train model on features
            total_train_loss += take_one_step(model, features, target)
            # if end of batch or end of dataset, validate model
            if steps % batch_size == 0 or steps == len(train_data)-1:
                valid_loss, lowest_valid_loss = common.validate_model(model, valid_data, lowest_valid_loss, filepath)

                if valid_loss < lowest_valid_loss:
                    lowest_valid_loss = valid_loss
                    common.save_model(model, filepath)

                total_valid_loss += valid_loss
                avg_valid_loss = total_valid_loss / (steps / batch_size)
                avg_train_loss = total_train_loss / steps
                print_batch_status(avg_train_loss, avg_valid_loss, start_time)
                prev_valid_losses.append(round(avg_valid_loss, 4))

                if terminate_early(prev_valid_losses):
                    print("\nTerminated epoch early due to stagnating or increasing validation loss.\n\n")
                    last_valid_loss = prev_valid_losses[-1]
                    break

        last_valid_loss = prev_valid_losses[-1]
        epoch += 1

        report = f"Time elapsed by epoch {epoch+1}: {round((time.time() - start_time)) / 60} mins."
        REPORTS.append(report)
        print(report)

    return last_valid_loss



#
# ------------- Find the Most Promising Models -----------------
#
def parameter_tuner(coin: str, model_architecture: str) -> None:
    data_aug_factor = 32
    batch_size = 256
    n_epochs = 5

    # ------------ DATA GENERATION ----------
    print("Creating datasets...")
    try:
        train_data, valid_data, test_data = common.get_datasets(coin, data_aug_factor)
    except:
        raise

    # ------------ MODEL TRAINING -----------
    model_number = 0
    for eta in np.arange(0.00025, 0.01025, 0.00025):
        for decay in np.arange(0.9999, 0.99999, 0.00001):
            for dropout in np.arange(0.05, 0.85, 0.05):
                print("Start of new Experiment\n__________________________")
                print(f"Model #{model_number}")
                print(f"Eta: {eta} | Decay: {decay} | Dropout: {dropout}")
                report = ""

                nn.set_model_parameters(dropout, eta, decay)
                nn.set_model(model_architecture)
                nn.set_model_props(nn.get_model())
                model = nn.get_model()

                # train model
                start_time = time.time()
                final_valid_loss = fully_train(model, (train_data, valid_data), start_time, f"models/{coin}_{model_architecture}_{model_number}_param_tuning.pt", n_epochs=n_epochs)

                # ------------ MODEL TESTING -----------
                # evaluate model
                model_acc = common.evaluate_model(model, test_data)
                model_acc_report = f"MODEL: {model_number}\nFinal Validation Loss: {final_valid_loss}\nPARAMETERS:\n\t{model_architecture}\n\teta: {nn.LEARNING_RATE} | decay: {nn.LEARNING_RATE_DECAY} | dropout: {nn.DROPOUT}\nDECISIONS:\n\tPerfect Decision: {model_acc[0]}\n\tTold to Hodl, though Should Have Bought/Sold: {model_acc[1]}\n\tSignal Should Have Been Hodl: {model_acc[2]}\n\tSignal and Answer Exact Opposite: {model_acc[3]}"
                report += model_acc_report
                print(model_acc_report)

                # ------------ RESULT HANDLING -----------
                # automatically save the best models to best as is
                if model_acc[0] > common.OUTSTANDING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
                    save_filepath = f"models/best/{coin}_{model_architecture}_{model_number}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
                    save_model(model, save_filepath)
                    with open(f"reports/{coin}_best_performers.txt", 'a') as f:
                        f.write(save_filepath + '\n')

                # save the model to the promising models folder
                if model_acc[0] > common.PROMISING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
                    save_filepath = f"models/promising/{coin}_{model_architecture}_{model_number}_param_tuning.pt"
                    common.save_model(model, save_filepath)

                # write the report independent of performance
                with open(f"reports/{coin}_Parameter_Tuning_Report_{model_architecture}.txt", "a") as f:
                    f.write(report + "\n\n")

                    print("Report written")

                model_number += 1



#
# -------------- Continue Training Most Successful Experiments --------------
#
def continue_training(coin: str, model_architecture: str) -> None:
    global REPORTS
    batch_size = 256
    n_epochs = 20

    # ------------ DATA GENERATION ----------
    start_time = time.time()
    data_aug_factor = 64
    print("Creating datasets...")
    try:
        train_data, valid_data, test_data = common.get_datasets(coin, data_aug_factor)
    except:
        raise

    print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")

    # ------------ MODEL TRAINING -----------
    promising_models = common.parse_training_reports(coin, model_architecture)
    for model_params in promising_models:
        model_filepath = f"models/promising/{coin}_{model_architecture}_{model_params['model_num']}_param_tuning.pt"
        model = common.load_model_by_params(model_filepath, model_params)

        dropout, eta, eta_decay = nn.get_model_parameters()
        reports = [f"Model: {model.get_class_name()}", f"Learning rate: {eta}", f"Learning rate decay: {eta_decay}", f"Chance of dropout: {dropout}", f"Batch size: {batch_size}", f"Epochs: {n_epochs}", f"Coin: {coin}"]

        start_time = time.time()
        print(f"Model #{model_params['model_num']}")

        fully_train(model, (train_data, valid_data), start_time, f"models/promising/{coin}_{model_architecture}_{model_params['model_num']}_lowest_val_loss.pt")

        # ------------ MODEL TESTING -----------
        # load model with lowest validation loss
        model = common.load_model(model, f"models/promising/{coin}_{model_architecture}_{model_params['model_num']}_lowest_val_loss.pt")
        report = "EVALUATE TRAINED MODEL"
        REPORTS.append(report)
        print(report)
        model_acc = common.evaluate_model(model, test_data)

        # ------------ RESULT HANDLING -----------
        # save iff accuracy is higher/lower than threshholds
        if model_acc[0] > model_params["accuracy"] and model_acc[3] < common.INACCURACY_THRESHOLD:
            save_filepath = f"models/best/{coin}_{model_architecture}_{model_params['model_num']}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
            common.save_model(model, save_filepath)
            with open(f"reports/{coin}_best_performers.txt", 'a') as f:
                f.write(save_filepath + '\n')

            save_report(coin, model_architecture)



def transfer_learner(coin: str) -> None:
    '''
    Train new models from models successfully trained on other cryptoasset datasets.
    '''
    # ------------ DATA GENERATION ----------
    # create the datasets
    start_time = time.time()
    data_aug_factor = 16
    print("Creating datasets...")
    try:
        train_data, valid_data, test_data = get_datasets(coin, data_aug_factor)
    except:
        raise
    print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")

    # ------------ MODEL TRAINING -----------
    # load previously trained model
    best_models = []
    with open("reports/bitcoin_best_performers.txt", 'r') as f:
        best_models = f.read().splitlines()

    # perform the transfer learning for each model
    for filename in best_models:
        model_params = common.get_model_params("bitcoin", filename)
        if model_params == None:
            print(f"{filename} Not Found. Continuing on to other models.")
            continue

        model = common.load_model_by_params(filename, model_params)

        # train on novel data
        dropout, eta, eta_decay = nn.get_model_parameters()
        reports = [f"Model: {model.get_class_name()}", f"Learning rate: {eta}", f"Learning rate decay: {eta_decay}", f"Chance of dropout: {dropout}"]

        start_time = time.time()
        print(f"Architecture: {filename}\nModel: {model_params['model_num']}")

        fully_train(model, (train_data, valid_data), start_time, f"models/promising/{coin}_{model_params['architecture']}_{model_params['model_num']}_lowest_val_loss.pt")

        # ------------ MODEL TESTING -----------
        # evaluate
        model = load_model(model, f"models/promising/{coin}_{model_params['architecture']}_{model_params['model_num']}_lowest_val_loss.pt")
        report = "EVALUATE TRAINED MODEL"
        REPORTS.append(report)
        print(report)
        model_acc = common.evaluate_model(model, test_data)

        # ------------ RESULT HANDLING -----------
        # save iff accuracy is higher/lower than threshholds
        if model_acc[0] > common.OUTSTANDING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
            save_filepath = f"models/best/{coin}_{model_params['architecture']}_{model_params['model_num']}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
            save_model(model, save_filepath)

            with open(f"reports/{coin}_best_performers.txt", 'a') as f:
                f.write(save_filepath + '\n')



#
# ------------ CONTROLLER METHOD ---------------
#
def fully_automated_training_pipeline() -> None:
    '''
    Pipeline involves three steps:

        1.) Parameter tune: 	find the most promising learning rates, decay rates, and dropout rates for the given architecture
        2.) Continue training:	take all the promising models found in the first step and give them more time to train
        3.) Cleanup:			delete all extraneous files created in the first two phases
    '''
    #neural_net_architecture = ["Pi_0", "Pi_1", "Pi_2", "Pi_3", "Pi_4", "Pi_5", "Pi_6", "Pi_7"]
    #neural_net_architecture = ["PC_0", "PC_1", "PC_2", "PC_3", "PC_4", "PC_5", "PC_6"]
    #neural_net_architecture = ["Laptop_0", "Laptop_1", "Laptop_2", "Laptop_3", "Laptop_4"]
    neural_net_architecture = ["Pi_4"]

    coin = "all"

    for model_architecture in neural_net_architecture:
        parameter_tuner(coin, model_architecture)
        continue_training(coin, model_architecture)
        cleanup(coin, "aggregate")




if __name__ == "__main__":
    try:
        fully_automated_training_pipeline()
        #transfer_learner("algorand")
    finally:
        print("Program terminated.")

'''
This file is used to train neural nets.
RUN: $ python3 -m utils.model_generation_engine.data_processor
'''
import os
import glob
import pandas as pd
import torch
import time
import numpy as np
from datetime import datetime
from typing import List, Tuple
from .. import common
from . import neural_nets as nn

#
# ------------ DELETING FUNCTIONS -----------
#
def cleanup(coin: str) -> None:
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
            os.rename(f, f"models/{coin}/{file_handle}")
            print(f"Successfully moved {f} to {coin}")
        except:
            print(f"Error when attempting to move {f}")

    # move best models to specified directory
    file_list = glob.glob(f"models/best/{coin}_*")
    for f in file_list:
        try:
            offset = len("models/best/")
            file_handle = f[offset:]
            os.rename(f, f"models/{coin}/{file_handle}")
            print(f"Successfully moved {f} to models/{coin}/")
        except:
            print(f"Error when attempting to move {f}")



#
# ---------- REPORTING FUNCTIONS ----------
#
def print_batch_status(model: nn.CryptoSoothsayer, avg_train_loss: float, avg_valid_loss:float, start_time: float) -> None:
    '''
    Prints summary for the latest batch to console.
    '''
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    report = f"System Time: {current_time} | Time Elapsed: {(time.time() - start_time) / 60:.1f} mins. | Avg. Training Loss: {avg_train_loss:.4f} | Avg. Validation Loss: {avg_valid_loss:.4f} | eta: {model.get_optimizer().state_dict()['param_groups'][0]['lr']:.6f}"
    print(report)



def make_and_save_list_of_best_performers(coin: str) -> None:
    # try removing previous version
    try:
        os.remove(f"reports/{coin}_best_performers.txt")
    except OSError:
        # file does not exist
        pass

    models = glob.glob(f"models/{coin}/{coin}*")

    with open(f"reports/{coin}_best_performers.txt", 'w') as f:
        for model in models:
            f.write(model + '\n')


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
    feature_tensor, target_tensor = common.convert_to_tensor(model, features, target)
    # Forward
    model_output = model(feature_tensor)
    loss = model.get_criterion()(model_output, target_tensor)
    # Backward
    model.get_optimizer().zero_grad()
    loss.backward()
    model.get_optimizer().step()
    # adjust learning rate
    model.get_scheduler().step()
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
                print_batch_status(model, avg_train_loss, avg_valid_loss, start_time)
                prev_valid_losses.append(round(avg_valid_loss, 4))

                if terminate_early(prev_valid_losses):
                    print("\nTerminated epoch early due to stagnating or increasing validation loss.\n\n")
                    last_valid_loss = prev_valid_losses[-1]
                    break

        last_valid_loss = prev_valid_losses[-1]
        epoch += 1

        report = f"Time elapsed by epoch {epoch+1}: {round((time.time() - start_time)) / 60} mins."
        print(report)

    return last_valid_loss



#
# ------------- Find the Most Promising Models -----------------
#
def parameter_tuner(coin: str, hidden_layer_size: int, data_aug_factor: int = 16) -> None:
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

                model = common.create_nn_model(hidden_layer_size, dropout, eta, decay)

                # train model
                start_time = time.time()
                final_valid_loss = fully_train(model, (train_data, valid_data), start_time, f"models/{coin}_{model.get_model_name()}_{model_number}_param_tuning.pt", n_epochs=n_epochs)

                # ------------ MODEL TESTING -----------
                # evaluate model
                model_acc = common.evaluate_model(model, test_data)
                model_acc_report = f"MODEL: {model_number}\nFinal Validation Loss: {final_valid_loss}\nPARAMETERS:\n\t{model.get_model_name()}\n\teta: {model.get_eta()} | decay: {model.get_eta_decay()} | dropout: {model.get_dropout().p}\nDECISIONS:\n\tPerfect Decision: {model_acc[0]}\n\tTold to Hodl, though Should Have Bought/Sold: {model_acc[1]}\n\tSignal Should Have Been Hodl: {model_acc[2]}\n\tSignal and Answer Exact Opposite: {model_acc[3]}"
                report += model_acc_report
                print(model_acc_report)

                # ------------ RESULT HANDLING -----------
                # automatically save the best models to best as is
                if model_acc[0] > common.OUTSTANDING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
                    save_filepath = f"models/best/{coin}_{model.get_model_name()}_{model_number}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
                    common.save_model(model, save_filepath)
                    with open(f"reports/{coin}_best_performers.txt", 'a') as f:
                        f.write(save_filepath + '\n')

                # save the model to the promising models folder
                if model_acc[0] > common.PROMISING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
                    save_filepath = f"models/promising/{coin}_{model.get_model_name()}_{model_number}_param_tuning.pt"
                    common.save_model(model, save_filepath)

                # write the report independent of performance
                with open(f"reports/{coin}_Parameter_Tuning_Report_{model.get_model_name()}.txt", "a") as f:
                    f.write(report + "\n\n")

                    print("Report written")

                model_number += 1



#
# -------------- Continue Training Most Successful Experiments --------------
#
def continue_training(coin: str, model_architecture: str, data_aug_factor: int = 32) -> None:
    batch_size = 256
    n_epochs = 20

    # ------------ DATA GENERATION ----------
    start_time = time.time()
    print("Creating datasets...")
    try:
        train_data, valid_data, test_data = common.get_datasets(coin, data_aug_factor)
    except:
        raise

    print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")

    # ------------ MODEL TRAINING -----------
    promising_models = common.parse_training_reports(coin, model_architecture)
    for model_params in promising_models:
        model_filepath = f"models/promising/{coin}_Hidden_{model_params['architecture']}_{model_params['model_num']}_param_tuning.pt"
        model = common.load_model_by_params(model_filepath, model_params)

        reports = [f"Model: {model.get_model_name()}", f"Learning rate: {model.get_eta()}", f"Learning rate decay: {model.get_eta_decay()}", f"Chance of dropout: {model.get_dropout().p}", f"Batch size: {batch_size}", f"Epochs: {n_epochs}", f"Coin: {coin}"]

        start_time = time.time()
        print(f"Model #{model_params['model_num']}")

        fully_train(model, (train_data, valid_data), start_time, f"models/promising/{coin}_{model_architecture}_{model_params['model_num']}_lowest_val_loss.pt")

        # ------------ MODEL TESTING -----------
        # load model with lowest validation loss
        model = common.load_model(model, f"models/promising/{coin}_{model_architecture}_{model_params['model_num']}_lowest_val_loss.pt")
        print("EVALUATE TRAINED MODEL")
        model_acc = common.evaluate_model(model, test_data)

        # ------------ RESULT HANDLING -----------
        # save iff accuracy is higher/lower than threshholds
        if model_acc[0] > model_params["accuracy"] and model_acc[3] < common.INACCURACY_THRESHOLD:
            save_filepath = f"models/best/{coin}_{model_architecture}_{model_params['model_num']}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
            common.save_model(model, save_filepath)
            with open(f"reports/{coin}_best_performers.txt", 'a') as f:
                f.write(save_filepath + '\n')



def transfer_learner(coin: str, data_aug_factor: int = 16) -> None:
    '''
    Train new models from models successfully trained on other cryptoasset datasets.
    '''
    # ------------ DATA GENERATION ----------
    # create the datasets
    start_time = time.time()
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
        reports = [f"Model: {model.get_model_name()}", f"Learning rate: {model.get_eta()}", f"Learning rate decay: {model.get_eta_decay()}", f"Chance of dropout: {model.get_dropout().p}"]

        start_time = time.time()
        print(f"Architecture: {filename}\nModel: {model_params['model_num']}")

        fully_train(model, (train_data, valid_data), start_time, f"models/promising/{coin}_{model_params['architecture']}_{model_params['model_num']}_lowest_val_loss.pt")

        # ------------ MODEL TESTING -----------
        # evaluate
        model = load_model(model, f"models/promising/{coin}_{model_params['architecture']}_{model_params['model_num']}_lowest_val_loss.pt")
        print("EVALUATE TRAINED MODEL")
        model_acc = common.evaluate_model(model, test_data)

        # ------------ RESULT HANDLING -----------
        # save iff accuracy is higher/lower than threshholds
        if model_acc[0] > common.OUTSTANDING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
            save_filepath = f"models/best/{coin}_{model_params['architecture']}_{model_params['model_num']}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
            common.save_model(model, save_filepath)

            with open(f"reports/{coin}_best_performers.txt", 'a') as f:
                f.write(save_filepath + '\n')




#
# ------------ CONTROLLER METHOD ---------------
#
def fully_automated_training_pipeline() -> None:
    '''
    Pipeline involves five steps:

        1.) Parameter tune:     find the most promising learning rates, decay rates, and dropout rates for the given architecture
        2.) Continue training:  take all the promising models found in the first step and give them more time to train
        3.) Cleanup:            delete all extraneous files created in the first two phases
        4.) Pruning models:     hold all models to more rigorous standards and keep only those that match
        5.) Make a list:        list all the best performers (for use in the signal_generator script)
    '''
    coin = "ethereum"
    layer_sizes = [19] #[x for x in range(nn.N_SIGNALS+2, nn.N_FEATURES)]

    for hidden_layer_size in layer_sizes:
        parameter_tuner(coin, hidden_layer_size, data_aug_factor=32)
        continue_training(coin, "Hidden_" + str(hidden_layer_size), data_aug_factor=64)
        cleanup(coin)
        common.prune_models_by_accuracy(coin)
        make_and_save_list_of_best_performers(coin)





if __name__ == "__main__":
    try:
        fully_automated_training_pipeline()
        #transfer_learner("algorand")
    finally:
        print("Program terminated.")

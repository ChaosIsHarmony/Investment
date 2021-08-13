'''
Houses all the variables and methods shared among files (DRY).
Houses interface for all sublibraries so that when modifying sublibrary, will not have to modify its method names in all other sub libraries.
'''
from .model_generation_engine import data_aggregator as dt_agg
from .model_generation_engine import data_preprocessor as dt_pp
from .model_generation_engine import data_processor as dt_p
from .model_generation_engine import dataset_methods as dt_m
from .model_generation_engine import model_methods as mm
from .model_generation_engine import neural_nets as nn
from . import risk_adjusted_return_calculator as rarc
import pandas as pd
import torch
from typing import List, Tuple

#
# ------------- CONSTANTS ------------
#
SIGNAL_FOR_N_DAYS_FROM_NOW = 7 * 5 # 7 * n weeks
PROMISING_ACCURACY_THRESHOLD = 0.655
OUTSTANDING_ACCURACY_THRESHOLD = 0.76
INACCURACY_THRESHOLD = 0.05
PRUNING_THRESHOLD_ALL = 0.6
PRUNING_THRESHOLD_VALID = 0.47
PRUNING_THRESHOLD_TEST = 0.7


coins = ["algorand", "bitcoin", "cardano", "chainlink", "ethereum", "polkadot", "solana"]
possible_coins = ["matic-network", "theta-token", "zilliqa"]

#
# ------------- INTERFACES ------------
#
# DATA AGGREGATOR
def aggregate_data_for_new_coins(coins: List[str], interval: int = 600) -> None:
    dt_agg.aggregate_data_for_new_coins(coins, interval)


def aggregate_new_data(coin: str, n_days: int) -> str:
    return dt_agg.fetch_missing_data_by_range(coin, n_days)



# DATA PREPROCESSOR
def clean_coin_data(coin: str, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    return dt_pp.clean_data(coin, data, start_date, end_date)


def handle_missing_data(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    return dt_pp.handle_missing_data(data, start_date, end_date)



# DATASET METHODS
def merge_newly_aggregated_data(coin: str, by_range: bool = True) -> str:
    return dt_m.merge_new_dataset_with_old(coin, by_range)


def merge_datasets(coin: str, datasets: List[pd.DataFrame], all_data: bool) -> None:
    dt_m.merge_datasets(coin, datasets, all_data)


def shuffle_data(data: List[Tuple[List[float],float]]) -> List[Tuple[List[float], float]]:
    return dt_m.shuffle_data(data)


def get_datasets(coin: str, data_aug_factor: int) -> Tuple[List[Tuple[List[float], float]], List[Tuple[List[float], float]], List[Tuple[List[float], float]]]:
    return dt_m.get_datasets(coin, data_aug_factor)


def load_data(coin: str) -> pd.DataFrame:
    return dt_m.load_data(coin)


def check_if_data_is_clean(data: pd.DataFrame) -> None:
    try:
        dt_m.check_if_data_is_clean(data)
    except:
        raise


def prepare_model_pruning_datasets(coin: str) -> Tuple[List[Tuple[List[float], float]], List[Tuple[List[float], float]], List[Tuple[List[float], float]]]:
    return dt_m.prepare_model_pruning_datasets(coin)


def convert_to_tensor(model: nn.CryptoSoothsayer, features: List[float], target: float) -> Tuple[torch.tensor, torch.tensor]:
    return dt_m.convert_to_tensor(model, features, target)



# MODEL METHODS
def save_model(model: nn.CryptoSoothsayer, filepath: str) -> None:
    mm.save_model(model, filepath)


def load_model(model: nn.CryptoSoothsayer, filepath: str) -> nn.CryptoSoothsayer:
    return mm.load_model(model, filepath)


def load_model_by_params(filepath: str, params: dict) -> nn.CryptoSoothsayer:
    return mm.load_model_by_params(filepath, params)


def load_pretrained_model(filepath: str) -> nn.CryptoSoothsayer:
    return mm.load_pretrained_model(filepath)


def evaluate_model(model: nn.CryptoSoothsayer, test_data: Tuple[List[float], float]) -> List[float]:
    return mm.evaluate_model(model, test_data)


def print_evaluation_status(model_accuracy: List[float]) -> str:
    return mm.print_evaluation_status(model_accuracy)


def validate_model(model: nn.CryptoSoothsayer, valid_data: List[Tuple[List[float], float]], lowest_valid_loss: float, filepath: str) -> Tuple[float, float]:
    return mm.validate_model(model, valid_data, lowest_valid_loss, filepath)


def prune_models_by_accuracy(coin: str) -> None:
    mm.prune_models_by_accuracy(coin)


def get_model_params(coin: str, filepath: str) -> dict:
    return mm.get_model_params(coin, filepath)


def parse_training_reports(coin: str, model_architecture: str) -> List[dict]:
    return mm.parse_reports(coin, model_architecture)



# NEURAL NETS
def create_nn_model(hidden_layer_size: int, dropout: float, eta: float, eta_decay: float) -> nn.CryptoSoothsayer:
    return nn.create_model(hidden_layer_size, dropout, eta, eta_decay)



# RISK ADJUSTED RETURN CALCULATOR
def get_sharpe_ratio(coin: str) -> float:
    return rarc.get_sharpe_ratio(coin)


def get_upi(coin: str) -> float:
    return rarc.get_upi(coin)

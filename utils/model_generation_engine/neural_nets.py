import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Tuple

# Fixed parameters
N_SIGNALS = 3
N_FEATURES = 23


class CryptoSoothsayer(nn.Module):
    def __init__(self, name: str, input_size: int, hidden_size: int, n_signals: int, dropout: float, eta: float, eta_decay: float):
        super(CryptoSoothsayer, self).__init__()
        # architecture
        self.name = name
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_output = nn.Linear(hidden_size, n_signals)
        # hyperparameters
        self.dropout = nn.Dropout(dropout)
        self.eta = eta
        self.eta_decay = eta_decay
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.eta)
        lambda1 = lambda epoch: self.eta_decay
        self.scheduler =  lr_scheduler.MultiplicativeLR(self.optimizer, lambda1)
        # device
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.to(self.device)


    def forward(self, inputs: torch.tensor) -> torch.tensor:
        out = self.dropout(F.relu(self.layer_1(inputs)))
        out = self.layer_output(out)
        return out


    def get_dropout(self):
        return self.dropout


    def get_eta(self):
        return self.eta


    def get_eta_decay(self):
        return self.eta_decay


    def get_criterion(self):
        return self.criterion


    def get_optimizer(self):
        return self.optimizer


    def get_scheduler(self):
        return self.scheduler


    def get_device(self):
        return self.device


    def get_model_name(self) -> str:
        return self.name

#
# -------------- GENERATOR ---------------
#
def create_model(hidden_layer_size: int, dropout: float, eta: float, eta_decay: float) -> CryptoSoothsayer:
    model = CryptoSoothsayer(f"Hidden_{hidden_layer_size}", N_FEATURES, hidden_layer_size, N_SIGNALS, dropout, eta, eta_decay)

    return model

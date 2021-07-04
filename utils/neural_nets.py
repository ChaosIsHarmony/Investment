import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Fixed parameters
N_SIGNALS = 5
N_SIGNALS_GRANULAR = 7
N_FEATURES = 25
# Tunable Hyperparameters
DROPOUT = None
LEARNING_RATE = None
LEARNING_RATE_DECAY = None
# Model
MODEL = None 
DEVICE = None
CRITERION = None
OPTIMIZER = None
SCHEDULER = None

#
# ---------- MODELS TRAINED ON RASPBERRY PI ----------
#
class CryptoSoothsayer_Pi_0(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Pi_0, self).__init__()
		self.layer_1 = nn.Linear(input_size, 2187)
		self.layer_2 = nn.Linear(2187, 729)
		self.layer_3 = nn.Linear(729, 243)
		self.layer_4 = nn.Linear(243, 81)
		self.layer_5 = nn.Linear(81, 243)
		self.layer_6 = nn.Linear(243, 81)
		self.layer_7 = nn.Linear(81, 27)
		self.layer_8 = nn.Linear(27, 9)
		self.layer_output = nn.Linear(9, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.dropout(F.relu(self.layer_4(out)))
		out = self.dropout(F.relu(self.layer_5(out)))
		out = self.dropout(F.relu(self.layer_6(out)))
		out = self.dropout(F.relu(self.layer_7(out)))
		out = self.dropout(F.relu(self.layer_8(out)))
		out = self.layer_output(out)
		return out


	def get_class_name(self):
		return "CryptoSoothsayer_Pi_0"



class CryptoSoothsayer_Pi_1(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Pi_1, self).__init__()
		self.layer_1 = nn.Linear(input_size, 243)
		self.layer_2 = nn.Linear(243, 729)
		self.layer_3 = nn.Linear(729, 2187)
		self.layer_4 = nn.Linear(2187, 729)
		self.layer_5 = nn.Linear(729, 243)
		self.layer_6 = nn.Linear(243, 81)
		self.layer_7 = nn.Linear(81, 243)
		self.layer_8 = nn.Linear(243, 81)
		self.layer_9 = nn.Linear(81, 27)
		self.layer_10 = nn.Linear(27, 9)
		self.layer_output = nn.Linear(9, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.dropout(F.relu(self.layer_4(out)))
		out = self.dropout(F.relu(self.layer_5(out)))
		out = self.dropout(F.relu(self.layer_6(out)))
		out = self.dropout(F.relu(self.layer_7(out)))
		out = self.dropout(F.relu(self.layer_8(out)))
		out = self.layer_output(out)
		return out


	def get_class_name(self):
		return "CryptoSoothsayer_Pi_1"



#
# ---------- MODELS TRAINED ON OLD PC ----------
#
class CryptoSoothsayer_PC_0(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_PC_0, self).__init__()
		self.layer_1 = nn.Linear(input_size, 15625)
		self.layer_2 = nn.Linear(15625, 3125)
		self.layer_3 = nn.Linear(3125, 625)
		self.layer_4 = nn.Linear(625, 125)
		self.layer_5 = nn.Linear(125, 25)
		self.layer_output = nn.Linear(25, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.dropout(F.relu(self.layer_4(out)))
		out = self.dropout(F.relu(self.layer_5(out)))
		out = self.layer_output(out)
		return out


	def get_class_name(self):
		return "CryptoSoothsayer_PC_0"



class CryptoSoothsayer_PC_1(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_PC_1, self).__init__()
		self.layer_1 = nn.Linear(input_size, 625)
		self.layer_2 = nn.Linear(625, 3125)
		self.layer_3 = nn.Linear(3125, 15625)
		self.layer_4 = nn.Linear(15625, 3125)
		self.layer_5 = nn.Linear(3125, 625)
		self.layer_6 = nn.Linear(625, 125)
		self.layer_7 = nn.Linear(125, 25)
		self.layer_output = nn.Linear(25, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.dropout(F.relu(self.layer_4(out)))
		out = self.dropout(F.relu(self.layer_5(out)))
		out = self.dropout(F.relu(self.layer_6(out)))
		out = self.dropout(F.relu(self.layer_7(out)))
		out = self.layer_output(out)
		return out


	def get_class_name(self):
		return "CryptoSoothsayer_PC_1"




#
# ---------- MODELS TRAINED ON LAPTOP ----------
#
class CryptoSoothsayer_Laptop_0(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Laptop_0, self).__init__()
		self.layer_1 = nn.Linear(input_size, 4096)
		self.layer_2 = nn.Linear(4096, 2048)
		self.layer_3 = nn.Linear(2048, 1024)
		self.layer_4 = nn.Linear(1024, 512)
		self.layer_5 = nn.Linear(512, 128)
		self.layer_6 = nn.Linear(128, 256)
		self.layer_7 = nn.Linear(256, 128)
		self.layer_8 = nn.Linear(128, 64)
		self.layer_9 = nn.Linear(64, 32)
		self.layer_10 = nn.Linear(32, 16)
		self.layer_11 = nn.Linear(16, 8)
		self.layer_output = nn.Linear(8, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.dropout(F.relu(self.layer_4(out)))
		out = self.dropout(F.relu(self.layer_5(out)))
		out = self.dropout(F.relu(self.layer_6(out)))
		out = self.dropout(F.relu(self.layer_7(out)))
		out = self.dropout(F.relu(self.layer_8(out)))
		out = self.dropout(F.relu(self.layer_9(out)))
		out = self.dropout(F.relu(self.layer_10(out)))
		out = self.dropout(F.relu(self.layer_11(out)))
		out = self.layer_output(out)
		return out

	
	def get_class_name(self):
		return "CryptoSoothsayer_Laptop_0"



class CryptoSoothsayer_Laptop_1(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Laptop_1, self).__init__()
		self.layer_1 = nn.Linear(input_size, 1024)
		self.layer_2 = nn.Linear(1024, 2048)
		self.layer_3 = nn.Linear(2048, 4096)
		self.layer_4 = nn.Linear(4096, 2048)
		self.layer_5 = nn.Linear(2048, 1024)
		self.layer_6 = nn.Linear(1024, 512)
		self.layer_7 = nn.Linear(512, 128)
		self.layer_8 = nn.Linear(128, 256)
		self.layer_9 = nn.Linear(256, 128)
		self.layer_10 = nn.Linear(128, 64)
		self.layer_11 = nn.Linear(64, 32)
		self.layer_12 = nn.Linear(32, 16)
		self.layer_13 = nn.Linear(16, 8)
		self.layer_output = nn.Linear(8, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.dropout(F.relu(self.layer_4(out)))
		out = self.dropout(F.relu(self.layer_5(out)))
		out = self.dropout(F.relu(self.layer_6(out)))
		out = self.dropout(F.relu(self.layer_7(out)))
		out = self.dropout(F.relu(self.layer_8(out)))
		out = self.dropout(F.relu(self.layer_9(out)))
		out = self.dropout(F.relu(self.layer_10(out)))
		out = self.dropout(F.relu(self.layer_11(out)))
		out = self.dropout(F.relu(self.layer_12(out)))
		out = self.dropout(F.relu(self.layer_13(out)))
		out = self.layer_output(out)
		return out

	
	def get_class_name(self):
		return "CryptoSoothsayer_Laptop_1"



class CryptoSoothsayer_Laptop_2(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Laptop_2, self).__init__()
		self.layer_1 = nn.Linear(input_size, 16384)
		self.layer_2 = nn.Linear(16384, 64)
		self.layer_output = nn.Linear(64, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.layer_output(out)
		return out

	
	def get_class_name(self):
		return "CryptoSoothsayer_Laptop_2"



class CryptoSoothsayer_Laptop_3(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Laptop_3, self).__init__()
		self.layer_1 = nn.Linear(input_size, 16384)
		self.layer_2 = nn.Linear(16384, 4096)
		self.layer_3 = nn.Linear(4096, 1024)
		self.layer_4 = nn.Linear(1024, 256)
		self.layer_5 = nn.Linear(256, 32)
		self.layer_output = nn.Linear(32, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.dropout(F.relu(self.layer_4(out)))
		out = self.dropout(F.relu(self.layer_5(out)))
		out = self.layer_output(out)
		return out

	
	def get_class_name(self):
		return "CryptoSoothsayer_Laptop_3"



class CryptoSoothsayer_Laptop_4(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Laptop_4, self).__init__()
		self.layer_1 = nn.Linear(input_size, 4096)
		self.layer_2 = nn.Linear(4096, 16384)
		self.layer_3 = nn.Linear(16384, 256)
		self.layer_output = nn.Linear(256, n_signals)
		self.dropout = nn.Dropout(DROPOUT)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.layer_output(out)
		return out

	
	def get_class_name(self):
		return "CryptoSoothsayer_Laptop_4"



#
# -------------- GETTERS & SETTERS ---------------
#

def set_model_props(model):
	global DEVICE, CRITERION, OPTIMIZER, SCHEDULER

	DEVICE = torch.device("cpu")
	model.to(DEVICE)
	CRITERION = nn.CrossEntropyLoss()
	OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	lambda1 = lambda epoch: LEARNING_RATE_DECAY 
	SCHEDULER =  lr_scheduler.MultiplicativeLR(OPTIMIZER, lambda1)



def set_model(model_architecture): 
	global MODEL

	if "Laptop_0" in model_architecture:
		MODEL = CryptoSoothsayer_Laptop_0(N_FEATURES, N_SIGNALS_GRANULAR)
	elif "Laptop_1" in model_architecture:
		MODEL = CryptoSoothsayer_Laptop_1(N_FEATURES, N_SIGNALS_GRANULAR)
	elif "Laptop_2" in model_architecture:
		MODEL = CryptoSoothsayer_Laptop_2(N_FEATURES, N_SIGNALS_GRANULAR)
	elif "Laptop_3" in model_architecture:
		MODEL = CryptoSoothsayer_Laptop_3(N_FEATURES, N_SIGNALS_GRANULAR)
	elif "Pi_0" in model_architecture:
		MODEL = CryptoSoothsayer_Pi_0(N_FEATURES, N_SIGNALS_GRANULAR)
	elif "Pi_1" in model_architecture:
		MODEL = CryptoSoothsayer_Pi_1(N_FEATURES, N_SIGNALS_GRANULAR)
	elif "PC_0" in model_architecture:
		MODEL = CryptoSoothsayer_PC_0(N_FEATURES, N_SIGNALS_GRANULAR)
	elif "PC_1" in model_architecture:
		MODEL = CryptoSoothsayer_PC_1(N_FEATURES, N_SIGNALS_GRANULAR)



def set_model_parameters(dropout=0, eta=0, eta_decay=0):
	global DROPOUT, LEARNING_RATE, LEARNING_RATE_DECAY

	DROPOUT = dropout
	LEARNING_RATE = eta
	LEARNING_RATE_DECAY = eta_decay



def set_pretrained_model(model):
	global MODEL
	MODEL = model



def get_model():
	global MODEL
	return MODEL



def get_model_parameters():
	global DROPOUT, LEARNING_RATE, LEARNING_RATE_DECAY
	return DROPOUT, LEARNING_RATE, LEARNING_RATE_DECAY



def get_device():
	global DEVICE
	return DEVICE

def get_criterion():
	global CRITERION
	return CRITERION

def get_optimizer():
	global OPTIMIZER
	return OPTIMIZER

def get_scheduler():
	global SCHEDULER
	return SCHEDULER


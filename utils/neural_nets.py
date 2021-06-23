import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Fixed parameters
N_SIGNALS = 5
N_FEATURES = 25
# Tunable Hyperparameters
DROPOUT = 0.25
LEARNING_RATE = 0.003
LEARNING_RATE_DECAY = 0.9999975

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
		out = self.layer_output(out)
		return out

	
	def get_class_name(self):
		return "CryptoSoothsayer_Laptop_1"




MODEL = CryptoSoothsayer_Laptop_0(N_FEATURES, N_SIGNALS)
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
lambda1 = lambda epoch: LEARNING_RATE_DECAY 
SCHEDULER =  lr_scheduler.MultiplicativeLR(OPTIMIZER, lambda1)

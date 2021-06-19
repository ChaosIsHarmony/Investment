import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


N_SIGNALS = 5
N_FEATURES = 26
DROPOUT = 0.15
LEARNING_RATE = 0.0015
LEARNING_RATE_DECAY = 0.99999

#
# ---------- MODELS TRAINED ON RASPBERRY PI ----------
#
class CryptoSoothsayer_Pi_0(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Pi_0, self).__init__()
		self.layer_1 = nn.Linear(input_size, 32)
		self.layer_2 = nn.Linear(32, 64)
		self.layer_3 = nn.Linear(64, 128)
		self.layer_4 = nn.Linear(128, 256)
		self.layer_5 = nn.Linear(256, 128)
		self.layer_6 = nn.Linear(128, 512)
		self.layer_7 = nn.Linear(512, 128)
		self.layer_8 = nn.Linear(128, 32)
		self.layer_output = nn.Linear(32, n_signals)
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
		self.layer_1 = nn.Linear(input_size, 64)
		self.layer_2 = nn.Linear(64, 128)
		self.layer_3 = nn.Linear(128, 256)
		self.layer_4 = nn.Linear(256, 512)
		self.layer_5 = nn.Linear(512, 1024)
		self.layer_6 = nn.Linear(1024, 512)
		self.layer_7 = nn.Linear(512, 256)
		self.layer_8 = nn.Linear(256, 64)
		self.layer_output = nn.Linear(64, n_signals)
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
		self.layer_1 = nn.Linear(input_size, 128)
		self.layer_2 = nn.Linear(128, 256)
		self.layer_3 = nn.Linear(256, 512)
		self.layer_4 = nn.Linear(512, 1024)
		self.layer_5 = nn.Linear(1024, 2048)
		self.layer_6 = nn.Linear(2048, 1024)
		self.layer_7 = nn.Linear(1024, 512)
		self.layer_8 = nn.Linear(512, 256)
		self.layer_9 = nn.Linear(256, 128)
		self.layer_10 = nn.Linear(128, 64)
		self.layer_11 = nn.Linear(64, 32)
		self.layer_output = nn.Linear(32, n_signals)
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
		return "CryptoSoothsayer_PC_0"


#
# ---------- MODELS TRAINED ON LAPTOP ----------
#
class CryptoSoothsayer_Laptop_0(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Laptop_0, self).__init__()
		self.layer_1 = nn.Linear(input_size, 256)
		self.layer_2 = nn.Linear(256, 1024)
		self.layer_3 = nn.Linear(1024, 512)
		self.layer_4 = nn.Linear(512, 1024)
		self.layer_5 = nn.Linear(1024, 256)
		self.layer_6 = nn.Linear(256, 512)
		self.layer_7 = nn.Linear(512, 128)
		self.layer_8 = nn.Linear(128, 256)
		self.layer_9 = nn.Linear(256, 128)
		self.layer_10 = nn.Linear(128, 64)
		self.layer_output = nn.Linear(64, n_signals)
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
		out = self.layer_output(out)
		return out

	
	def get_class_name(self):
		return "CryptoSoothsayer_Laptop_0"



class CryptoSoothsayer_Laptop_1(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer_Laptop_1, self).__init__()
		self.layer_1 = nn.Linear(input_size, 256)
		self.layer_2 = nn.Linear(256, 1024)
		self.layer_3 = nn.Linear(1024, 512)
		self.layer_4 = nn.Linear(512, 256)
		self.layer_5 = nn.Linear(256, 1024)
		self.layer_6 = nn.Linear(1024, 512)
		self.layer_7 = nn.Linear(512, 256)
		self.layer_8 = nn.Linear(256, 1024)
		self.layer_9 = nn.Linear(1024, 512)
		self.layer_10 = nn.Linear(512, 256)
		self.layer_11 = nn.Linear(256, 128)
		self.layer_12 = nn.Linear(128, 64)
		self.layer_13 = nn.Linear(64, 32)
		self.layer_output = nn.Linear(32, n_signals)
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
		self.layer_1 = nn.Linear(input_size, 256)
		self.layer_2 = nn.Linear(256, 2048)
		self.layer_3 = nn.Linear(2048, 4096)
		self.layer_4 = nn.Linear(4096, 256)
		self.layer_5 = nn.Linear(256, 1024)
		self.layer_6 = nn.Linear(1024, 512)
		self.layer_7 = nn.Linear(512, 256)
		self.layer_8 = nn.Linear(256, 128)
		self.layer_9 = nn.Linear(128, 64)
		self.layer_10 = nn.Linear(64, 32)
		self.layer_output = nn.Linear(32, n_signals)
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
		out = self.layer_output(out)
		return out

	
	def get_class_name(self):
		return "CryptoSoothsayer_Laptop_2"



MODEL = CryptoSoothsayer_Laptop_0(N_FEATURES, N_SIGNALS)
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
lambda1 = lambda epoch: LEARNING_RATE_DECAY 
SCHEDULER =  lr_scheduler.MultiplicativeLR(OPTIMIZER, lambda1)

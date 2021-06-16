import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


N_SIGNALS = 5
N_FEATURES = 18
DROPOUT = 0.25
LEARNING_RATE = 0.001


class CryptoSoothsayerBin_0(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayerBin_0, self).__init__()
		self.layer_1 = nn.Linear(input_size, 32)
		self.layer_2 = nn.Linear(32, 64)
		self.layer_3 = nn.Linear(64, 128)
		self.layer_4 = nn.Linear(128, 256)
		self.layer_5 = nn.Linear(256, 512)
		self.layer_6 = nn.Linear(512, 1024)
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



class CryptoSoothsayerBin_1(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayerBin_1, self).__init__()
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



class CryptoSoothsayerBin_2(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayerBin_2, self).__init__()
		self.layer_1 = nn.Linear(input_size, 2048)
		self.layer_2 = nn.Linear(2048, 1024)
		self.layer_3 = nn.Linear(1024, 512)
		self.layer_4 = nn.Linear(512, 256)
		self.layer_5 = nn.Linear(256, 128)
		self.layer_6 = nn.Linear(128, 64)
		self.layer_7 = nn.Linear(64, 32)
		self.layer_8 = nn.Linear(32, 16)
		self.layer_9 = nn.Linear(16, 8)
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
		out = self.layer_output(out)
		return out



class CryptoSoothsayerBin_3(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayerBin_3, self).__init__()
		self.layer_1 = nn.Linear(input_size, 128)
		self.layer_2 = nn.Linear(128, 512)
		self.layer_3 = nn.Linear(512, 128)
		self.layer_4 = nn.Linear(128, 512)
		self.layer_5 = nn.Linear(512, 128)
		self.layer_6 = nn.Linear(128, 256)
		self.layer_7 = nn.Linear(256, 64)
		self.layer_8 = nn.Linear(64, 128)
		self.layer_9 = nn.Linear(128, 32)
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
		out = self.layer_output(out)
		return out



class CryptoSoothsayerDec_0(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayerDec_0, self).__init__()
		self.layer_1 = nn.Linear(input_size, 100)
		self.layer_2 = nn.Linear(100, 250)
		self.layer_3 = nn.Linear(250, 500)
		self.layer_4 = nn.Linear(500, 1000)
		self.layer_5 = nn.Linear(1000, 500)
		self.layer_6 = nn.Linear(500, 250)
		self.layer_7 = nn.Linear(250, 100)
		self.layer_8 = nn.Linear(100, 50)
		self.layer_9 = nn.Linear(50, 25)
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
		out = self.dropout(F.relu(self.layer_8(out)))
		out = self.dropout(F.relu(self.layer_9(out)))
		out = self.layer_output(out)
		return out 


MODEL = CryptoSoothsayerBin_3(N_FEATURES, N_SIGNALS)
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
lambda1 = lambda epoch: 0.99999
SCHEDULER =  lr_scheduler.MultiplicativeLR(OPTIMIZER, lambda1)



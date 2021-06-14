import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import random

# correspond to signal column scale from 0-4
DECISIONS = ["BUY 2X", "BUY X", "HODL", "SELL Y", "SELL 2Y"]
N_SIGNALS = 5
N_FEATURES = 17
BATCH_SIZE = 7
EPOCHS = 10
LEARNING_RATE = 0.005
MODEL_FILEPATH = "models/model.pt"
MODEL_CHECKPOINT_FILEPATH = "models/model_checkpoint.pt"

coin = "bitcoin"

# Load data
data = pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv")
data = data.drop(columns=["Unnamed: 0", "date"])
data["signal"] = data["signal"].astype("int64")

# Split into training, testing
# 85-15 split
n_datapoints = data.shape[0]
train_end = int(round(n_datapoints*0.85))

def generate_dataset(data, limit, offset, data_aug_per_sample=0):
	'''
	NOTE: data_aug_per_sample param determines how many extra datapoints to generate per each original datapoint.
	'''
	dataset = []
	for row in range(offset, limit):
		row_features = []
		for feature in range(N_FEATURES):
			row_features.append(data.iloc[row, feature])
		datapoint_tuple = (row_features, data.iloc[row, -1])
		dataset.append(datapoint_tuple)

		for i in range(data_aug_per_sample):
			row_features_aug = []
			for feature in range(N_FEATURES):
				rand_factor = 1 + random.uniform(-0.00001, 0.0001)
				row_features_aug.append(data.iloc[row, feature] * rand_factor)
			datapoint_tuple = (row_features_aug, data.iloc[row, -1])
			dataset.append(datapoint_tuple)

	return dataset


train_data = generate_dataset(data, train_end, 0, 10)
print(f"Length Training Data: {len(train_data)}")

test_data = generate_dataset(data, n_datapoints, train_end, 0)
print(f"Length Testing Data: {len(test_data)}") 

# NN
class CryptoSoothsayer(nn.Module):
	def __init__(self, input_size, n_signals):
		super(CryptoSoothsayer, self).__init__()
		self.layer_1 = nn.Linear(input_size, 128)
		self.layer_2 = nn.Linear(128, 256)
		self.layer_2b = nn.Linear(256, 512)
		self.layer_2c = nn.Linear(512, 1024)
		self.layer_2d = nn.Linear(1024, 2048)
		self.layer_2e = nn.Linear(2048, 1024)
		self.layer_2f = nn.Linear(1024, 512)
		self.layer_2g = nn.Linear(512, 256)
		self.layer_3 = nn.Linear(256, 128)
		self.layer_4 = nn.Linear(128, 64)
		self.layer_5 = nn.Linear(64, 32)
		self.layer_output = nn.Linear(32, n_signals)
		self.dropout = nn.Dropout(0.35)


	def forward(self, inputs):
		out = self.dropout(F.relu(self.layer_1(inputs)))
		out = self.dropout(F.relu(self.layer_2(out)))
		out = self.dropout(F.relu(self.layer_2b(out)))
		out = self.dropout(F.relu(self.layer_2c(out)))
		out = self.dropout(F.relu(self.layer_2d(out)))
		out = self.dropout(F.relu(self.layer_2e(out)))
		out = self.dropout(F.relu(self.layer_2f(out)))
		out = self.dropout(F.relu(self.layer_2g(out)))
		out = self.dropout(F.relu(self.layer_3(out)))
		out = self.dropout(F.relu(self.layer_4(out)))
		out = self.dropout(F.relu(self.layer_5(out)))
		out = self.layer_output(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs


# save model
def save_checkpoint():
	checkpoint = {
		"model": model,
		"state_dict": model.state_dict(),
		"epochs": EPOCHS,
		"average_loss": average_loss,
		"device": device,
		"optimizer_state": optimizer.state_dict(),
		"batch_size": BATCH_SIZE
	}
	torch.save(checkpoint, MODEL_CHECKPOINT_FILEPATH)

def save_model_state():
	torch.save(model.state_dict(), MODEL_FILEPATH)



# load model
def load_checkpoint():
	checkpoint = torch.load(MODEL_CHECKPOINT_FILEPATH)
	model = checkpoint["model"]
	model.optimizer_state = checkpoint["optimizer_state"]
	model.load_state_dict(checkpoint["state_dict"])
	model.device = checkpoint["device"]
	model.average_loss = checkpoint["average_loss"]
	
	return model

def load_model():
	model = CryptoSoothsayer(N_FEATURES, N_SIGNALS)
	model.load_state_dict(torch.load(MODEL_FILEPATH))
	
	return model


# Time sequence, so maybe shuffle is inappropriate
# ACTUALLY: okay, because of SMAs providing historical info
def shuffle_data(data):
	size = len(data)
	for row_ind in range(size):
		swap_row_ind = random.randrange(size)
		tmp_row = data[swap_row_ind]
		data[swap_row_ind] = data[row_ind]
		data[row_ind] = tmp_row

	return data


def train(model, data, epochs):
	for epoch in range(epochs):
		steps = 0
		print_every = 100
		running_loss = 0
		data = shuffle_data(data)

		for feature, target in data:
			model.train()
			feature_tensor = torch.tensor([feature], dtype=torch.float32)
			feature_tensor = feature_tensor.to(device)
			target_tensor = torch.tensor([target], dtype=torch.int64)
			target_tensor = target_tensor.to(device)
			# Forward
			log_probs = model(feature_tensor)
			loss = criterion(log_probs, target_tensor)
			# Bacward
			model.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			running_loss += loss.item()
			steps += 1

			if steps % print_every == 0:
				model.eval()
				cur_avg_loss = running_loss/print_every
				average_loss.append(cur_avg_loss)
				print(f"Epoch: {epoch+1} / {epochs} | Training Loss: {cur_avg_loss:.4f} | eta: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
				running_loss = 0

		print(f"Time elapsed by epoch {epoch+1}: {round((time.time() - start_time)) / 60} mins.")

	return model


def train_and_save(model, train_data, epochs):
	# Train
	model = train(model, train_data, epochs)
	print(f"Total training time: {round((time.time() - start_time)) / 60} mins.")

	# Save
	save_checkpoint()


model = CryptoSoothsayer(N_FEATURES, N_SIGNALS)
device = torch.device("cpu")
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
lambda1 = lambda epoch: 0.99999
scheduler =  lr_scheduler.MultiplicativeLR(optimizer, lambda1)
start_time = time.time()
average_loss = []

# Training
train_and_save(model, train_data, EPOCHS)

# Load
model = load_checkpoint()
model.eval()

correct = 0
for feature, target in test_data:
	feature_tensor = torch.tensor([feature], dtype=torch.float32)
	target_tensor = torch.tensor([target], dtype=torch.int64)

	with torch.no_grad():
		output = model(feature_tensor)

	decision = torch.argmax(output, dim=1)

	if decision == target_tensor:
		correct += 1


print(f"Model accuracy: {correct/len(test_data)}")

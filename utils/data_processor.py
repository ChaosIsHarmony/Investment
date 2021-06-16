import pandas as pd
import torch
import time
import random
import neural_nets as nn
'''
WHEN testing need this version instead
from utils import neural_nets as nn
'''

# correspond to signal column scale from 0-4
DECISIONS = ["BUY 2X", "BUY X", "HODL", "SELL Y", "SELL 2Y"]
BATCH_SIZE = 7
EPOCHS = 30 
BIN = 0
DEC = 1
MODEL = "models/model.pt"
MODEL_CHECKPOINT = "models/model_checkpoint_63.pt"
DEVICE = torch.device("cpu")
COIN = "bitcoin"
AVERAGE_LOSS = []

#
# ------------ DATA RELATED -----------
#
def generate_dataset(data, limit, offset, data_aug_per_sample=0):
	'''
	NOTES: 
		- data_aug_per_sample param determines how many extra datapoints to generate per each original datapoint.
		- signal_ratios variable is used to upsample underrepresented categories more than their counterparts when augmenting the data
	'''
	# to determine relative frequency of signals
	new_data = data.iloc[:limit,:]
	vals = new_data["signal"].value_counts().sort_index()
	signal_ratios = [vals.max()/x for x in vals]

	dataset = []
	for row in range(offset, limit):
		target = data.iloc[row, -1]
		row_features = []

		for feature in range(nn.N_FEATURES):
			row_features.append(data.iloc[row, feature])
		datapoint_tuple = (row_features, target)
		dataset.append(datapoint_tuple)

		# this evens out the datapoints per category
		for i in range(data_aug_per_sample * round(signal_ratios[target])):
			row_features_aug = []
			for feature in range(nn.N_FEATURES):
				rand_factor = 1 + random.uniform(-0.00001, 0.00001)
				row_features_aug.append(data.iloc[row, feature] * rand_factor)
			datapoint_tuple_aug = (row_features_aug, target)
			dataset.append(datapoint_tuple_aug)

	return dataset


def get_datasets():
	# Load data
	data = pd.read_csv(f"datasets/complete/{COIN}_historical_data_complete.csv")
	data = data.drop(columns=["Unnamed: 0", "date"])
	data["signal"] = data["signal"].astype("int64")

	# Split into training, testing
	# 85-15 split
	n_datapoints = data.shape[0]
	train_end = int(round(n_datapoints*0.85))


	train_data = generate_dataset(data, train_end, 0, 20)
	print(f"Length Training Data: {len(train_data)}")

	test_data = generate_dataset(data, n_datapoints, train_end, 0)
	print(f"Length Testing Data: {len(test_data)}") 

	return train_data, test_data


#
# ------------ HELPER FUNCTIONS -----------
#
# save model
def save_checkpoint(filepath, model):
	checkpoint = {
		"model": model,
		"state_dict": model.state_dict(),
		"epochs": EPOCHS,
		"average_loss": AVERAGE_LOSS,
		"device": DEVICE,
		"optimizer_state": nn.OPTIMIZER.state_dict(),
		"batch_size": BATCH_SIZE,
		"dropout": nn.DROPOUT
	}
	torch.save(checkpoint, filepath)


def save_model_state(filepath):
	torch.save(model.state_dict(), filepath)


# load model
def load_checkpoint(filepath):
	checkpoint = torch.load(filepath)
	model = checkpoint["model"]
	model.optimizer_state = checkpoint["optimizer_state"]
	model.load_state_dict(checkpoint["state_dict"])
	model.device = checkpoint["device"]
	model.average_loss = checkpoint["average_loss"]
	
	return model


def load_model(neural_net, filepath):
	model = neural_net
	model.load_state_dict(torch.load(filepath))

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


def train(model, data, epochs, start_time):
	for epoch in range(epochs):
		steps = 0
		print_every = 1000
		running_loss = 0
		data = shuffle_data(data)

		for feature, target in data:
			model.train()
			feature_tensor = torch.tensor([feature], dtype=torch.float32)
			feature_tensor = feature_tensor.to(DEVICE)
			target_tensor = torch.tensor([target], dtype=torch.int64)
			target_tensor = target_tensor.to(DEVICE)
			# Forward
			log_probs = model(feature_tensor)
			loss = nn.CRITERION(log_probs, target_tensor)
			# Backward
			model.zero_grad()
			loss.backward()
			nn.OPTIMIZER.step()
			nn.SCHEDULER.step()
			running_loss += loss.item()
			steps += 1

			if steps % print_every == 0:
				model.eval()
				cur_avg_loss = running_loss/print_every
				AVERAGE_LOSS.append(cur_avg_loss)
				print(f"Epoch: {epoch+1} / {epochs} | Training Loss: {cur_avg_loss:.4f} | eta: {nn.OPTIMIZER.state_dict()['param_groups'][0]['lr']:.6f}")
				running_loss = 0

		print(f"Time elapsed by epoch {epoch+1}: {round((time.time() - start_time)) / 60} mins.")

	return model


def train_and_save(model, train_data, epochs, filepath, start_time):
	# Train
	model = train(model, train_data, epochs, start_time)
	print(f"Total training time: {round((time.time() - start_time)) / 60} mins.")

	# Save
	save_checkpoint(filepath, model)


def run():
	# get data
	train_data, test_data = get_datasets()

	#
	# ------------ MODEL TRAINING -----------
	#
	model = nn.MODEL
	model.to(DEVICE)
	start_time = time.time()

	# Training
	#train_and_save(model, train_data, EPOCHS, MODEL_CHECKPOINT, start_time)

	#
	# ------------ MODEL TESTING -----------
	#
	# Load
	model = load_checkpoint(MODEL_CHECKPOINT)
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


run()

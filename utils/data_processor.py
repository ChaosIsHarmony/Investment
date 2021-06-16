import pandas as pd
import torch
import time
import random
import neural_nets as nn
import numpy as np
'''
WHEN testing need this version instead
from utils import neural_nets as nn
'''

# correspond to signal column scale from 0-4
DECISIONS = ["BUY 2X", "BUY X", "HODL", "SELL Y", "SELL 2Y"]
BATCH_SIZE = 512 
EPOCHS = 2 
MODEL = "models/model.pt"
MODEL_CHECKPOINT = "models/model_checkpoint.pt"
DEVICE = torch.device("cpu")
COIN = "bitcoin"

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

	# Split into training, validation, testing
	# 70-15-15 split
	n_datapoints = data.shape[0]
	train_end = int(round(n_datapoints*0.7))
	valid_end = train_end + int(round(n_datapoints*0.15))


	train_data = generate_dataset(data, train_end, 0, 16)
	print(f"Length Training Data: {len(train_data)}")

	valid_data = generate_dataset(data, train_end, 0, 4)
	print(f"Length Validation Data: {len(valid_data)}")

	test_data = generate_dataset(data, n_datapoints, valid_end, 0)
	print(f"Length Testing Data: {len(test_data)}") 

	return train_data, valid_data, test_data


#
# ------------ HELPER FUNCTIONS -----------
#
# save model
def save_checkpoint(filepath, model):
	checkpoint = {
		"model": model,
		"state_dict": model.state_dict(),
		"epochs": EPOCHS,
		"device": DEVICE,
		"optimizer_state": nn.OPTIMIZER.state_dict(),
		"batch_size": BATCH_SIZE,
		"dropout": nn.DROPOUT
	}
	torch.save(checkpoint, filepath)


def save_model_state(filepath, model):
	torch.save(model.state_dict(), filepath)


# load model
def load_checkpoint(filepath):
	checkpoint = torch.load(filepath)
	model = checkpoint["model"]
	model.optimizer_state = checkpoint["optimizer_state"]
	model.load_state_dict(checkpoint["state_dict"])
	model.device = checkpoint["device"]
	
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


def train(model, train_data, valid_data, start_time):
	min_valid_loss = np.inf
	
	for epoch in range(EPOCHS):
		steps = 0
		train_loss = 0.0
		train_data = shuffle_data(train_data)
		valid_data = shuffle_data(valid_data)

		for feature, target in train_data:
			steps += 1
			model.train()
			# make data pytorch compatible
			feature_tensor = torch.tensor([feature], dtype=torch.float32)
			feature_tensor = feature_tensor.to(DEVICE)
			target_tensor = torch.tensor([target], dtype=torch.int64)
			target_tensor = target_tensor.to(DEVICE)
			# Forward
			model_output = model(feature_tensor)
			loss = nn.CRITERION(model_output, target_tensor)
			# Backward
			nn.OPTIMIZER.zero_grad()
			loss.backward()
			nn.OPTIMIZER.step()
			# adjust learning rate
			nn.SCHEDULER.step()
			train_loss += loss.item() * feature_tensor.size(0)
			print(f"FT size(0): {feature_tensor.size(0)}")

			#validation
			if steps % BATCH_SIZE == 0 or steps == len(train_data)-1:
				model.eval()
				valid_loss = 0.0
				for feature, target in valid_data:
					# make data pytorch compatible
					feature_tensor = torch.tensor([feature], dtype=torch.float32)
					feature_tensor = feature_tensor.to(DEVICE)
					target_tensor = torch.tensor([target], dtype=torch.int64)
					target_tensor = target_tensor.to(DEVICE)
					# model makes prediction
					model_output = model(feature_tensor)
					loss = nn.CRITERION(model_output, target_tensor)
					valid_loss = loss.item() * feature_tensor.size(0)

				if valid_loss < min_valid_loss:
					min_valid_loss = valid_loss
					save_model_state(MODEL, model)
					print(f"Epoch: {epoch+1} / {EPOCHS} | Training Loss: {train_loss/steps:.4f} | Validation Loss: {min_valid_loss:.4f} | eta: {nn.OPTIMIZER.state_dict()['param_groups'][0]['lr']:.6f}")
				running_loss = 0

		print(f"Time elapsed by epoch {epoch+1}: {round((time.time() - start_time)) / 60} mins.")

	return model


def train_and_save(model, train_data, valid_data, start_time):
	# Train
	model = train(model, train_data, valid_data, start_time)

	# Save
	save_checkpoint(MODEL_CHECKPOINT, model)


def evaluate_model(model, test_data):
	model.eval()
	correct = 0
	mostly_correct = 0
	normal_fail = 0
	nasty_fail = 0
	catastrophic_fail = 0
	for feature, target in test_data:
		feature_tensor = torch.tensor([feature], dtype=torch.float32)
		target_tensor = torch.tensor([target], dtype=torch.int64)

		with torch.no_grad():
			output = model(feature_tensor)

		decision = torch.argmax(output, dim=1)
	
		if decision == target_tensor:
			correct += 1
		elif (target_tensor == 0 or target_tensor == 1) and (decision == 0 or decision == 1):
			mostly_correct += 1
		elif (target_tensor == 3 or target_tensor == 4) and (decision == 3 or decision == 4):
			mostly_correct += 1
		elif (target_tensor > 2 and decision < 2) or (target_tensor < 2 and decision > 2):
			catastrophic_fail += 1
		elif target_tensor == 2 and (decision == 0 or decision == 4):
			nasty_fail += 1
		else:
			normal_fail += 1

	print(f"Model perfect accuracy: {correct/len(test_data)}")
	print(f"Model good enough accuracy: {(mostly_correct + correct)/len(test_data)}")
	print(f"Model normal fail rate: {normal_fail/len(test_data)}")
	print(f"Model nasty fail rate: {nasty_fail/len(test_data)}")
	print(f"Model catastrophic fail rate: {catastrophic_fail/len(test_data)}")



def run():
	# 
	# ------------ DATA GENERATION ----------
	#
	train_data, valid_data, test_data = get_datasets()

	#
	# ------------ MODEL TRAINING -----------
	#
	model = nn.MODEL
	model.to(DEVICE)
	start_time = time.time()

	# Training
#	train_and_save(model, train_data, valid_data, start_time)

	#
	# ------------ MODEL TESTING -----------
	#
	# Load
	model = load_checkpoint(MODEL_CHECKPOINT)
#	model = load_model(nn.MODEL, MODEL)
	evaluate_model(model, test_data)
	
	


run()

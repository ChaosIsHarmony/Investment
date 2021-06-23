import os
import pandas as pd
import torch
import time
import random
import numpy as np
import neural_nets as nn
'''
WHEN testing need this version instead
'''
#from utils import neural_nets as nn


DEVICE = torch.device("cpu")
nn.MODEL.to(DEVICE)
MODEL_FILEPATH = f"models/{nn.MODEL.get_class_name()}.pt"
MODEL_CHECKPOINT_FILEPATH = f"models/checkpoint_{nn.MODEL.get_class_name()}.pt"
BATCH_SIZE = 256 
EPOCHS = 1 
COIN = "bitcoin"
REPORTS = [f"Model: {nn.MODEL.get_class_name()}", f"Learning rate: {nn.LEARNING_RATE}", f"Learning rate decay: {nn.LEARNING_RATE_DECAY}", f"Chance of dropout: {nn.DROPOUT}", f"Batch size: {BATCH_SIZE}", f"Epochs: {EPOCHS}", f"Coin: {COIN}"]

#
# ------------ DATA RELATED -----------
#
def shuffle_data(data):
	'''
	Used for shuffling the data during the training/validation phases.
	'''
	size = len(data)
	for row_ind in range(size):
		swap_row_ind = random.randrange(size)
		tmp_row = data[swap_row_ind]
		data[swap_row_ind] = data[row_ind]
		data[row_ind] = tmp_row

	return data



def generate_dataset(data, limit, offset, data_aug_per_sample=0):
	'''
	Returns a list of tuples, of which the first element of the tuple is the list of values for the features and the second is the target value
	NOTES: 
		- data_aug_per_sample param determines how many extra datapoints to generate per each original datapoint * its frequency metric (i.e., signal_ratios)
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


def get_datasets(coin, data_aug_factor=16):
	# Load data
	data = pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv")
	data = data.drop(columns=["date"])
	data["signal"] = data["signal"].astype("int64")

	# Split into training, validation, testing
	# 70-15-15 split
	n_datapoints = data.shape[0]
	train_end = int(round(n_datapoints*0.7))
	valid_end = train_end + int(round(n_datapoints*0.15))


	train_data = generate_dataset(data, train_end, 0, data_aug_factor)
	REPORTS.append(f"Length Training Data: {len(train_data)}")

	valid_data = generate_dataset(data, valid_end, train_end)
	REPORTS.append(f"Length Validation Data: {len(valid_data)}")

	test_data = generate_dataset(data, n_datapoints, valid_end)
	REPORTS.append(f"Length Testing Data: {len(test_data)}") 

	return train_data, valid_data, test_data


#
# ------------ SAVING/LOADING FUNCTIONS -----------
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


#
# ----------- TRAINING FUNCTIONS ----------
#
def convert_to_tensor(feature, target):
	feature_tensor = torch.tensor([feature], dtype=torch.float32)
	feature_tensor = feature_tensor.to(DEVICE)
	target_tensor = torch.tensor([target], dtype=torch.int64)
	target_tensor = target_tensor.to(DEVICE)

	return feature_tensor, target_tensor



def take_one_step(feature, target, train_loss):
	# set to train mode here to activate components like dropout
	nn.MODEL.train()
	# make data pytorch compatible
	feature_tensor, target_tensor = convert_to_tensor(feature, target)
	# Forward
	model_output = nn.MODEL(feature_tensor)
	loss = nn.CRITERION(model_output, target_tensor)
	# Backward
	nn.OPTIMIZER.zero_grad()
	loss.backward()
	nn.OPTIMIZER.step()
	# adjust learning rate
	nn.SCHEDULER.step()
	train_loss += loss.item()

	return train_loss



def validate_model(valid_data, train_loss, min_valid_loss):
	# set to evaluate mode to turn off components like dropout
	nn.MODEL.eval()
	valid_loss = 0.0
	for feature, target in valid_data:
		# make data pytorch compatible
		feature_tensor, target_tensor = convert_to_tensor(feature, target)
		# model makes prediction
		with torch.no_grad():
			model_output = nn.MODEL(feature_tensor)
			loss = nn.CRITERION(model_output, target_tensor)
			valid_loss += loss.item()

	if valid_loss/len(valid_data) < min_valid_loss:
		min_valid_loss = valid_loss/len(valid_data)
		save_model_state(MODEL_FILEPATH, nn.MODEL)
		report = f"Training Loss: {train_loss:.4f} | Validation Loss: {min_valid_loss:.4f} | eta: {nn.OPTIMIZER.state_dict()['param_groups'][0]['lr']:.6f}"
		REPORTS.append(report)
		print(report)

	return min_valid_loss



def train(train_data, valid_data, start_time):
	min_valid_loss = np.inf 
	
	train_data = shuffle_data(train_data)

	for epoch in range(EPOCHS):
		steps = 0
		train_loss = 0.0

		for feature, target in train_data:
			steps += 1
			# train model on feature
			train_loss = take_one_step(feature, target, train_loss)
			# if end of batch or end of dataset, validate model
			if steps % BATCH_SIZE == 0 or steps == len(train_data)-1:
				min_valid_loss = validate_model(valid_data, train_loss/steps, min_valid_loss)

		report = f"Time elapsed by epoch {epoch+1}: {round((time.time() - start_time)) / 60} mins."
		REPORTS.append(report)
		print(report)


def train_and_save(train_data, valid_data, start_time):
	# Train
	train(train_data, valid_data, start_time)

	# Save
	save_checkpoint(MODEL_CHECKPOINT_FILEPATH, nn.MODEL)



def evaluate_model(model, test_data):
	model.eval()
	correct = 0
	mostly_correct = 0
	safe_fail = 0
	nasty_fail = 0
	catastrophic_fail = 0
	for feature, target in test_data:
		feature_tensor, target_tensor = convert_to_tensor(feature, target)

		with torch.no_grad():
			output = model(feature_tensor)

		decision = torch.argmax(output, dim=1)

		# flawless
		if decision == target_tensor:
			correct += 1
		# correct direction, but extent was wrong (e.g., target = BUY X, decision = BUY 2X)
		elif (target_tensor == 0 or target_tensor == 1) and (decision == 0 or decision == 1):
			mostly_correct += 1
		# correct direction, but extent was wrong (e.g., target = SELL X, decision = SELL 2X)
		elif (target_tensor == 3 or target_tensor == 4) and (decision == 3 or decision == 4):
			mostly_correct += 1
		# catastrophic failure (e.g., told to buy when should have sold)
		elif (target_tensor > 2 and decision < 2) or (target_tensor < 2 and decision > 2):
			catastrophic_fail += 1
		# severe failure (e.g., should have hodled but was told to buy or sell
		elif target_tensor == 2 and (decision < 2 or decision > 2):
			nasty_fail += 1
		# decision was to hodl, but should have sold or bought
		else:
			safe_fail += 1

	report = f"""
	POSITIVE:
		[++] Perfect accuracy: {correct/len(test_data):>10.4f}
		[+] Model good enough accuracy: {(mostly_correct + correct)/len(test_data):>10.4f}
	NEGATIVE:
		[-] Told to hodl but should have sold/bought rate: {safe_fail/len(test_data):>10.4f}
		[--] Should have hodled but told to sell/buy rate: {nasty_fail/len(test_data):>10.4f}
		[---] Told to do the opposite of correct move rate: {catastrophic_fail/len(test_data):>10.4f}
		"""
	REPORTS.append(report)
	print(report)

	return [correct/len(test_data), (mostly_correct+correct)/len(test_data), nasty_fail/len(test_data), catastrophic_fail/len(test_data)]

#
# ---------- REPORTING FUNCTIONS ----------
#
def join_all_reports():
	final_report = ""
	for report in REPORTS:
		final_report += report + "\n"
	return final_report



def generate_report():
	report = open(f"reports/{nn.MODEL.get_class_name()}_report.txt", "w")
	report.write(join_all_reports())
	report.close()


#
# _____________________________________
#
def run():
	# 
	# ------------ DATA GENERATION ----------
	#
	train_data, valid_data, test_data = get_datasets(COIN)

	#
	# ------------ MODEL TRAINING -----------
	#
	start_time = time.time()
	train_and_save(train_data, valid_data, start_time)

	#
	# ------------ MODEL TESTING -----------
	#
	# Load
	model_checkpoint = load_checkpoint(MODEL_CHECKPOINT_FILEPATH)
	model = load_model(nn.MODEL, MODEL_FILEPATH)
	report = "EVALUATE FULLY TRAINED MODEL"
	REPORTS.append(report)
	print(report)
	evaluate_model(model_checkpoint, test_data)
	report = "EVALUATE VALIDATION-BASED MODEL"
	REPORTS.append(report)
	print(report)
	evaluate_model(model, test_data)

	#
	# ---------- GENERATE REPORT -----------
	#
	generate_report()
	


def validate_model_param_tuning(valid_data, min_valid_loss, model_architecture, model_counter):
	# set to evaluate mode to turn off components like dropout
	nn.MODEL.eval()
	valid_loss = 0.0
	for feature, target in valid_data:
		# make data pytorch compatible
		feature_tensor, target_tensor = convert_to_tensor(feature, target)
		# model makes prediction
		with torch.no_grad():
			model_output = nn.MODEL(feature_tensor)
			loss = nn.CRITERION(model_output, target_tensor)
			valid_loss += loss.item()

	if valid_loss/len(valid_data) < min_valid_loss:
		min_valid_loss = valid_loss/len(valid_data)
		save_model_state(f"models/CS_{model_architecture}_{model_counter}_mod.pt", nn.MODEL)

	return min_valid_loss




def parameter_tuner():
	train_data, valid_data, test_data = get_datasets(COIN)
	best = [0, 0, 1, 1]
	model_counter = 0

	for eta in np.arange(0.001, 0.005, 0.0005):
		nn.LEARNING_RATE = eta
		for decay in np.arange(0.9999, 0.99999, 0.00001):	
			nn.LEARNING_RATE_DECAY = decay
			for dropout in np.arange(0.05, 0.6, 0.05):
				report = "" 
				nn.DROPOUT = dropout
				
				model_architecture = "Laptop_0"
				nn.MODEL = nn.CryptoSoothsayer_Laptop_0(nn.N_FEATURES, nn.N_SIGNALS) 
				start_time = time.time()
				# Train
				min_valid_loss = np.inf 
				train_data = shuffle_data(train_data)

				for epoch in range(EPOCHS):
					steps = 0
					train_loss = 0.0

					for feature, target in train_data:
						steps += 1
						# train model on feature
						train_loss = take_one_step(feature, target, train_loss)
						# if end of batch or end of dataset, validate model
						if steps % BATCH_SIZE == 0 or steps == len(train_data)-1:
							min_valid_loss = validate_model_param_tuning(valid_data, min_valid_loss, model_architecture, model_counter)

					print(f"Time Elapsed: {(time.time() - start_time) / 60} mins. | Training Loss: {train_loss/steps} | Min Validation Loss: {min_valid_loss}")

				# Save
				save_checkpoint(f"models/CS_{model_architecture}_{model_counter}_chk.pt", nn.MODEL)


				model_checkpoint = load_checkpoint(f"models/CS_{model_architecture}_{model_counter}_chk.pt")
				model = load_model(nn.MODEL, f"models/CS_{model_architecture}_{model_counter}_mod.pt")
			
				chk_pt_acc = evaluate_model(model_checkpoint, test_data)
				mod_acc = evaluate_model(model, test_data)

				if chk_pt_acc[1] >= best[1] or chk_pt_acc[3] <= best[3]:
					report += f"CHECKPOINT FOR MODEL: {model_counter}\nPARAMETERS:\n\tLaptop_0\n\teta: {eta} | decay: {decay} | dropout: {dropout}\nDECISIONS:\n\tPerfect Decision: {chk_pt_acc[0]}\n\tAcceptable Decision: {chk_pt_acc[1]}\n\tSignal Should Have Been Hodl: {chk_pt_acc[2]}\n\tSignal and Answer Exact Opposite: {chk_pt_acc[3]}\n"
					save_model_state(f"models/CS_{model_architecture}_{model_counter}_0.pt", nn.MODEL)

				if mod_acc[1] >= best[1] or mod_acc[3] <= best[3]:
					report += f"MODEL: {model_counter}\nPARAMETERS:\n\tLaptop_0\n\teta: {eta} | decay: {decay} | dropout: {dropout}\nDECISIONS:\n\tPerfect Decision: {mod_acc[0]}\n\tAcceptable Decision: {mod_acc[1]}\n\tSignal Should Have Been Hodl: {mod_acc[2]}\n\tSignal and Answer Exact Opposite: {mod_acc[3]}"
					save_model_state(f"models/CS_{model_architecture}_{model_counter}_1.pt", nn.MODEL)
				
				# erase chkpt and model
				os.remove(f"models/CS_{model_architecture}_{model_counter}_chk.pt")
				os.remove(f"models/CS_{model_architecture}_{model_counter}_mod.pt")
				
				if len(report) > 0:
					with open(f"reports/Parameter_Tuning_Report.txt", "a") as f:
					# starting from index 1 to avoid first triple space divider
						f.write(report + "\n\n")

					print("Report written")

				model_counter += 1

parameter_tuner()
#run()

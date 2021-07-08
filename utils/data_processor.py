'''
This file is used to train the neural nets. It is comprised of several trainers:

	- The Parameter Tuner
		- Used to find promising parameters to more fully train networks on
	- The Transfer Learner
		- Used to continue the learning either in parameter tuned models or when training for other cryptoassets (change COIN variable to reflect dataset)
'''
import os
import pandas as pd
import torch
from datetime import datetime
import time
import random
import numpy as np
import param_trainer_parser as ptp
import neural_nets as nn
'''
WHEN testing need these versions instead
'''
#from utils import neural_nets as nn
#from utils import param_trainer_parser

BATCH_SIZE = 256 
EPOCHS = 20
COIN = "bitcoin"
REPORTS = []

#
# ------------ DATA RELATED -----------
#
def shuffle_data(data):
	'''
	Used for shuffling the data during the training/validation phases.
	NOTE: Param data is a Python list.
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

		# -1 excludes the signal column
		for feature in range(data.shape[1] - 1):
			row_features.append(data.iloc[row, feature])
		datapoint_tuple = (row_features, target)
		dataset.append(datapoint_tuple)

		# this evens out the datapoints per category
		for i in range(data_aug_per_sample * round(signal_ratios[target])):
			row_features_aug = []
			# -1 excludes the signal column
			for feature in range(data.shape[1] - 1):
				rand_factor = 1 + random.uniform(-0.000001, 0.000001)
				row_features_aug.append(data.iloc[row, feature] * rand_factor)
			datapoint_tuple_aug = (row_features_aug, target)
			dataset.append(datapoint_tuple_aug)

	return dataset


def get_datasets(coin, data_aug_factor=0):
	'''
	Splits dataset into training, validation, and testing datasets.
	NOTE: uses no data augmentation by default and will only apply data_aug_factor to the training dataset.
	'''
	global REPORTS


	# Load data
	data = pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv")
	data = data.drop(columns=["date"])
	data["signal"] = data["signal"].astype("int64")

	# Check for any anomalous, unnormalized data in all columns except the signal column
	for c in range(len(data.columns)-1):
		for r in range(len(data)):
			assert (data.iloc[r, c] > 1) == False, f"ERROR: Unnormalized data still present in the dataset in column: {data.columns[c]}."

	# Split into training, validation, testing
	# 70-15-15 split
	n_datapoints = data.shape[0]
	train_end = int(round(n_datapoints*0.7))
	valid_end = train_end + int(round(n_datapoints*0.15))


	train_data = generate_dataset(data, train_end, 0, data_aug_factor)
	REPORTS.append(f"Length Training Data: {len(train_data)}")
	print("Training dataset created.")

	valid_data = generate_dataset(data, valid_end, train_end)
	REPORTS.append(f"Length Validation Data: {len(valid_data)}")
	print("Validation dataset created.")

	test_data = generate_dataset(data, n_datapoints, valid_end)
	REPORTS.append(f"Length Testing Data: {len(test_data)}") 
	print("Testing dataset created.")

	return train_data, valid_data, test_data


#
# ------------ SAVING/LOADING FUNCTIONS -----------
#
# save model
def save_model(model, filepath):
	torch.save(model.state_dict(), filepath)


# load model
def load_model(neural_net, filepath):
	model = neural_net
	model.load_state_dict(torch.load(filepath))

	return model


#
# ----------- TRAINING FUNCTIONS ----------
#
def convert_to_tensor(feature, target):
	'''
	Converts the feature vector and target into pytorch-compatible tensors.
	'''
	feature_tensor = torch.tensor([feature], dtype=torch.float32)
	feature_tensor = feature_tensor.to(nn.get_device())
	target_tensor = torch.tensor([target], dtype=torch.int64)
	target_tensor = target_tensor.to(nn.get_device())

	return feature_tensor, target_tensor



def take_one_step(model, feature, target):
	'''
	Forward propogates a single feature vector through the network, the back propogates based on the loss.
	Returns the training loss.
	'''
	# set to train mode here to activate components like dropout
	model.train()
	# make data pytorch compatible
	feature_tensor, target_tensor = convert_to_tensor(feature, target)
	# Forward
	model_output = model(feature_tensor)
	loss = nn.get_criterion()(model_output, target_tensor)
	# Backward
	nn.get_optimizer().zero_grad()
	loss.backward()
	nn.get_optimizer().step()
	# adjust learning rate
	nn.get_scheduler().step()
	train_loss = loss.item()

	return train_loss



def validate_model(model, valid_data, lowest_valid_loss, filepath):
	'''
	Validates the model on the validation dataset.
	Saves model if validation loss is lower than the current lowest.
	Returns the average validation loss and lowest validation loss.
	'''
	# set to evaluate mode to turn off components like dropout
	model.eval()
	valid_loss = 0.0
	for feature, target in valid_data:
		# make data pytorch compatible
		feature_tensor, target_tensor = convert_to_tensor(feature, target)
		# model makes prediction
		with torch.no_grad():
			model_output = model(feature_tensor)
			loss = nn.get_criterion()(model_output, target_tensor)
			valid_loss += loss.item()

	if valid_loss/len(valid_data) < lowest_valid_loss:
		lowest_valid_loss = valid_loss/len(valid_data)
		save_model(model, filepath)

	return valid_loss/len(valid_data), lowest_valid_loss



def terminate_early(prev_valid_losses):
	'''
	Sends signal to terminate early if the validation loss is increasing over a 10-batch interval.
	'''
	if len(prev_valid_losses) >= 10:
		ind = len(prev_valid_losses) - 1
		valid_loss_trend = 0

		while ind > 0:
			valid_loss_trend += prev_valid_losses[ind] - prev_valid_losses[ind-1]
			ind -= 1
		
		if valid_loss_trend >= 0:
			return True
				
		del prev_valid_losses[0]

	return False



def print_batch_status(avg_train_loss, avg_valid_loss, start_time):
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	report = f"System Time: {current_time} | Time Elapsed: {(time.time() - start_time) / 60:.1f} mins. | Avg. Training Loss: {avg_train_loss:.4f} | Avg. Validation Loss: {avg_valid_loss:.4f} | eta: {nn.OPTIMIZER.state_dict()['param_groups'][0]['lr']:.6f}"
	REPORTS.append(report)
	print(report)



def fully_train(model, train_data, valid_data, start_time, filepath):
	global REPORTS, EPOCHS, BATCH_SIZE

	epoch = 0
	lowest_valid_loss = np.inf

	while epoch < EPOCHS:
		# setup
		train_data = shuffle_data(train_data)
		steps = 0
		prev_valid_losses = []
		total_train_loss = 0.0
		total_valid_loss = 0.0

		for feature, target in train_data:
			steps += 1
			# train model on feature
			total_train_loss += take_one_step(model, feature, target)
			# if end of batch or end of dataset, validate model
			if steps % BATCH_SIZE == 0 or steps == len(train_data)-1:
				valid_loss, lowest_valid_loss = validate_model(model, valid_data, lowest_valid_loss, filepath)

				total_valid_loss += valid_loss
				avg_valid_loss = total_valid_loss / (steps / BATCH_SIZE)
				avg_train_loss = total_train_loss / steps
				print_batch_status(avg_train_loss, avg_valid_loss, start_time)
				prev_valid_losses.append(round(avg_valid_loss, 4))
				if terminate_early(prev_valid_losses):
					print("\nTerminated epoch early due to stagnating or increasing validation loss.\n\n")
					break
	
		epoch += 1

		report = f"Time elapsed by epoch {epoch+1}: {round((time.time() - start_time)) / 60} mins."
		REPORTS.append(report)
		print(report)



def evaluate_model(model, test_data):
	global REPORTS

	model.eval()
	correct = 0
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
		# catastrophic failure (e.g., told to buy when should have sold)
		elif (target_tensor > 1 and decision < 1) or (target_tensor < 1 and decision > 1):
			catastrophic_fail += 1
		# severe failure (e.g., should have hodled but was told to buy or sell
		elif target_tensor == 1 and (decision < 1 or decision > 1):
			nasty_fail += 1
		# decision was to hodl, but should have sold or bought
		else:
			safe_fail += 1

	report = f"""
	POSITIVE:
		[+] Perfect accuracy: {correct/len(test_data):>10.4f}
	NEGATIVE:
		[-] Told to hodl but should have sold/bought rate: {safe_fail/len(test_data):>10.4f}
		[--] Should have hodled but told to sell/buy rate: {nasty_fail/len(test_data):>10.4f}
		[---] Told to do the opposite of correct move rate: {catastrophic_fail/len(test_data):>10.4f}
		"""
	REPORTS.append(report)
	print(report)

	return [correct/len(test_data), safe_fail/len(test_data), nasty_fail/len(test_data), catastrophic_fail/len(test_data)]



#
# ---------- REPORTING FUNCTIONS ----------
#
def join_all_reports():
	global REPORTS

	final_report = ""
	for report in REPORTS:
		final_report += report + "\n"
	return final_report



def generate_report():
	with open(f"reports/{nn.get_model().get_class_name()}_report.txt", "w") as report:
		report.write(join_all_reports())



#
# ------------- Find the Most Promising Models -----------------
#

def parameter_tuner():
	global COIN, BATCH_SIZE, EPOCHS
	data_aug_factor = 32
	print("Creating datasets...")
	train_data, valid_data, test_data = get_datasets(COIN, data_aug_factor)
	model_counter = 0

	for eta in np.arange(0.00025, 0.01025, 0.00025):
		for decay in np.arange(0.9999, 0.99999, 0.00001):	
			for dropout in np.arange(0.05, 0.85, 0.05):
				print("Start of new Experiment\n__________________________")
				print(f"Eta: {eta} | Decay: {decay} | Dropout: {dropout}")
				report = "" 
				
				model_architecture = "Laptop_1"
				nn.set_model_parameters(dropout, eta, decay)
				nn.set_model(model_architecture) 
				nn.set_model_props(nn.get_model())
				model = nn.get_model()

				start_time = time.time()
				lowest_valid_loss = np.inf
				prev_valid_losses = [] 
				prev_train_losses = [] 

				# Train
				for epoch in range(5):
					# setup
					train_data = shuffle_data(train_data)
					steps = 0
					total_train_loss = 0.0
					total_valid_loss = 0.0

					for feature, target in train_data:
						steps += 1
						# train model on feature
						total_train_loss += take_one_step(model, feature, target)
						# if end of batch or end of dataset, validate model
						if steps % BATCH_SIZE == 0 or steps == len(train_data)-1:
							valid_loss, lowest_valid_loss = validate_model(model, valid_data, lowest_valid_loss, f"models/{COIN}_{model_architecture}_{model_counter}_param_tuning.pt")
							total_valid_loss += valid_loss

							avg_valid_loss = total_valid_loss / (steps / BATCH_SIZE)
							avg_train_loss = total_train_loss / steps
							
							print_batch_status(avg_train_loss, avg_valid_loss, start_time)
							
							prev_train_losses.append(avg_train_loss)
							prev_valid_losses.append(round(avg_valid_loss, 4))
							
							if terminate_early(prev_valid_losses):
								print("\nTerminated epoch early due to stagnating or increasing validation loss.\n\n")
								break

				
				mod_acc = evaluate_model(model, test_data)
				
				report += f"MODEL: {model_counter}\nLast Training Loss: {prev_train_losses[-1]} | Last Valid Loss: {prev_valid_losses[-1]}\nPARAMETERS:\n\t{model_architecture}\n\teta: {nn.LEARNING_RATE} | decay: {nn.LEARNING_RATE_DECAY} | dropout: {nn.DROPOUT}\nDECISIONS:\n\tPerfect Decision: {mod_acc[0]}\n\tTold to Hodl, though Should Have Bought/Sold: {mod_acc[1]}\n\tSignal Should Have Been Hodl: {mod_acc[2]}\n\tSignal and Answer Exact Opposite: {mod_acc[3]}"
				
				if len(report) > 0:
					with open(f"reports/Parameter_Tuning_Report_{model_architecture}.txt", "a") as f:
					# starting from index 1 to avoid first triple space divider
						f.write(report + "\n\n")

					print("Report written")

				model_counter += 1

#parameter_tuner()



#
# -------------- Continue Training Most Successful Experiments --------------
#
def continue_training():
	global REPORTS, COIN, EPOCHS, BATCH_SIZE

	# 
	# ------------ DATA GENERATION ----------
	#
	start_time = time.time()
	data_aug_factor = 64
	print("Creating datasets...")
	train_data, valid_data, test_data = get_datasets(COIN, data_aug_factor)
	print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")

	#
	# ------------ MODEL TRAINING -----------
	#
	promising_models = ptp.parse_reports("Laptop_0")
	for model_params in promising_models:
		model_architecture = model_params["architecture"]
		model_number = model_params["model_num"]
		model_filepath = f"models/{COIN}_{model_architecture}_{model_number}_param_tuning.pt"
	
		nn.set_model_parameters(dropout = model_params["dropout"], eta = model_params["eta"], eta_decay = model_params["decay"])
		nn.set_model(model_architecture)
		nn.set_pretrained_model(load_model(nn.get_model(), model_filepath))
		nn.set_model_props(nn.get_model())
		model = nn.get_model()

		dropout, eta, eta_decay = nn.get_model_parameters()
		reports = [f"Model: {model.get_class_name()}", f"Learning rate: {eta}", f"Learning rate decay: {eta_decay}", f"Chance of dropout: {dropout}", f"Batch size: {BATCH_SIZE}", f"Epochs: {EPOCHS}", f"Coin: {COIN}"]

		start_time = time.time()
	
		fully_train(model, train_data, valid_data, start_time, f"models/{COIN}_{model_architecture}_{model_number}_lowest_val_loss.pt")

		#
		# ------------ MODEL TESTING -----------
		#
		# load model with lowest validation loss
		model = load_model(model, f"models/{COIN}_{model_architecture}_{model_number}_lowest_val_loss.pt")
		report = "EVALUATE TRAINED MODEL"
		REPORTS.append(report)
		print(report)
		model_acc = evaluate_model(model, test_data)

		# save iff accuracy is higher/lower than threshholds
		if model_acc[0] > ptp.ACCURACY_THRESHOLD and model_acc[3] < ptp.INACCURACY_THRESHOLD:
			save_model(model, f"models/{COIN}_{model_architecture}_{model_number}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt")

			#
			# ---------- GENERATE REPORT -----------
			#
			generate_report()
	

continue_training()

def try_model():
		model_number = 2817
		train_data, valid_data, test_data = get_datasets("bitcoin", 0)
		nn.set_model_parameters(dropout=0 , eta=0, eta_decay=0)
		model = load_model(nn.CryptoSoothsayer_Laptop_0(nn.N_FEATURES, nn.N_SIGNALS), f"models/bitcoin_Laptop_0_{model_number}_lowest_val_loss.pt")
		model_acc = evaluate_model(model, test_data)
	


def transfer_learner():
	# load data

	# load previously trained model

	# train on novel data

	# evaluate
	pass



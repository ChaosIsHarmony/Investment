'''
This file is used to train the neural nets. It is comprised of several trainers:

	- The Parameter Tuner
		- Used to find promising parameters to more fully train networks on
	- The Transfer Learner
		- Used to continue the learning either in parameter tuned models or when training for other cryptoassets (change COIN variable to reflect dataset)
'''
import os
import glob
import pandas as pd
import torch
from datetime import datetime
import time
import random
import numpy as np
import common
import neural_nets as nn
'''
WHEN testing need these versions instead
'''
#from utils import neural_nets as nn
#from utils import param_trainer_parser

BATCH_SIZE = 256 
EPOCHS = 20
COIN = "all"
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



def load_data(coin):
	'''
	Loads relevant data for given coin.
	'''
	data = pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv")
	data = data.drop(columns=["date"])
	data["signal"] = data["signal"].astype("int64")

	return data



def is_data_clean(data):
	'''
	Checks for any anomalous, unnormalized data in all columns except the signal column.
	'''
	for c in range(len(data.columns)-1):
		for r in range(len(data)):
			if (data.iloc[r, c] > 1):
				raise Exception(f"Data unfit for processing! Unnormalized data still present in the dataset in column = {data.columns[c]}, row = {r}.")



def get_datasets(coin, data_aug_factor=0):
	'''
	Splits dataset into training, validation, and testing datasets.
	NOTE: uses no data augmentation by default and will only apply data_aug_factor to the training dataset.
	'''
	global REPORTS

	data = load_data(coin)
	try:
		is_data_clean(data)
	except:
		raise

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
# ------------ MODEL SAVING/LOADING/DELETING FUNCTIONS -----------
#
def save_model(model, filepath):
	torch.save(model.state_dict(), filepath)



def load_model(neural_net, filepath):
	model = neural_net
	model.load_state_dict(torch.load(filepath))

	return model



def cleanup():
	'''
	Delete all the *.pt files from models that are not worth keeping.
	'''
	# delete all non-promising models
	file_list = glob.glob("models/*.pt")
	for f in file_list:
		try:
			os.remove(f)
		except:
			print(f"Error when attempting to remove {f}.")
	
	# delete trained promising models lowest validation that didn't make the cut
	# NOTE: we are still keeping the promising models to train later with different data aug factors
	file_list = glob.glob("models/promising/*lowest_val_loss.pt")
	for f in file_list:
		try:
			os.remove(f)
		except:
			print(f"Error when attempting to remove {f}.")



#
# ---------- REPORTING FUNCTIONS ----------
#
def join_all_reports():
	'''
	Collects all reports for different steps of the training process into a single string in prep for saving to disk.
	'''
	global REPORTS

	final_report = ""
	for report in REPORTS:
		final_report += report + "\n"
	return final_report



def save_report(coin, architecture):
	with open(f"reports/{coin}_{architecture}_report.txt", "w") as report:
		report.write(join_all_reports())



def print_batch_status(avg_train_loss, avg_valid_loss, start_time):
	'''
	Prints summary for the latest batch to console and appends it to the global REPORTS variable
	'''
	global REPORTS

	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	report = f"System Time: {current_time} | Time Elapsed: {(time.time() - start_time) / 60:.1f} mins. | Avg. Training Loss: {avg_train_loss:.4f} | Avg. Validation Loss: {avg_valid_loss:.4f} | eta: {nn.OPTIMIZER.state_dict()['param_groups'][0]['lr']:.6f}"
	REPORTS.append(report)
	print(report)


def print_evaluation_status(model_accuracy):
	'''
	Prints summary for model evaluation.
	'''
	global REPORTS

	report = f"""
	POSITIVE:
		[+] Perfect accuracy: {model_accuracy[0]:>10.4f}
	NEGATIVE:
		[-] Told to hodl but should have sold/bought rate: {model_accuracy[1]:>10.4f}
		[--] Should have hodled but told to sell/buy rate: {model_accuracy[2]:>10.4f}
		[---] Told to do the opposite of correct move rate: {model_accuracy[3]:>10.4f}
		"""
	REPORTS.append(report)
	print(report)


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



def evaluate_model(model, test_data, verbose=True):
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

	model_accuracy = [correct/len(test_data), safe_fail/len(test_data), nasty_fail/len(test_data), catastrophic_fail/len(test_data)]

	if verbose:
		print_evaluation_status(model_accuracy)

	return model_accuracy 


#
# ------------- Find the Most Promising Models -----------------
#

def parameter_tuner(model_architecture):
	global COIN, BATCH_SIZE, EPOCHS
	data_aug_factor = 32
	print("Creating datasets...")
	try:
		train_data, valid_data, test_data = get_datasets(COIN, data_aug_factor)
	except:
		raise
	model_number = 0

	for eta in np.arange(0.00025, 0.01025, 0.00025):
		for decay in np.arange(0.9999, 0.99999, 0.00001):	
			for dropout in np.arange(0.05, 0.85, 0.05):
				print("Start of new Experiment\n__________________________")
				print(f"Model #{model_number}")
				print(f"Eta: {eta} | Decay: {decay} | Dropout: {dropout}")
				report = "" 
				
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
							valid_loss, lowest_valid_loss = validate_model(model, valid_data, lowest_valid_loss, f"models/{COIN}_{model_architecture}_{model_number}_param_tuning.pt")
							total_valid_loss += valid_loss

							avg_valid_loss = total_valid_loss / (steps / BATCH_SIZE)
							avg_train_loss = total_train_loss / steps
							
							print_batch_status(avg_train_loss, avg_valid_loss, start_time)
							
							prev_train_losses.append(avg_train_loss)
							prev_valid_losses.append(round(avg_valid_loss, 4))
							
							if terminate_early(prev_valid_losses):
								print("\nTerminated epoch early due to stagnating or increasing validation loss.\n\n")
								break

				
				model_acc = evaluate_model(model, test_data)
				
				report += f"MODEL: {model_number}\nLast Training Loss: {prev_train_losses[-1]} | Last Valid Loss: {prev_valid_losses[-1]}\nPARAMETERS:\n\t{model_architecture}\n\teta: {nn.LEARNING_RATE} | decay: {nn.LEARNING_RATE_DECAY} | dropout: {nn.DROPOUT}\nDECISIONS:\n\tPerfect Decision: {model_acc[0]}\n\tTold to Hodl, though Should Have Bought/Sold: {model_acc[1]}\n\tSignal Should Have Been Hodl: {model_acc[2]}\n\tSignal and Answer Exact Opposite: {model_acc[3]}"
				
				# automatically save the best models to best as is
				if model_acc[0] > common.OUTSTANDING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
					save_filepath = f"models/best/{COIN}_{model_architecture}_{model_number}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
					save_model(model, save_filepath)
					with open(f"reports/{COIN}_best_performers.txt", 'a') as f:
						f.write(save_filepath + '\n')
				# save the model to the promising models folder
				# saves to both locations for further training
				if model_acc[0] > common.PROMISING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
					save_filepath = f"models/promising/{COIN}_{model_architecture}_{model_number}_param_tuning.pt"
					save_model(model, save_filepath)

				# write the report on all models independent of performance
				if len(report) > 0:
					with open(f"reports/{COIN}_Parameter_Tuning_Report_{model_architecture}.txt", "a") as f:
					# starting from index 1 to avoid first triple space divider
						f.write(report + "\n\n")

					print("Report written")

				model_number += 1



	

#
# -------------- Continue Training Most Successful Experiments --------------
#
def load_model_by_params(model_filepath, model_params):
	nn.set_model_parameters(dropout = model_params["dropout"], eta = model_params["eta"], eta_decay = model_params["decay"])
	nn.set_model(model_params["architecture"])
	nn.set_pretrained_model(load_model(nn.get_model(), model_filepath))
	nn.set_model_props(nn.get_model())
	return nn.get_model()


	
def continue_training(model_architecture):
	global REPORTS, COIN, EPOCHS, BATCH_SIZE

	# 
	# ------------ DATA GENERATION ----------
	#
	start_time = time.time()
	data_aug_factor = 64
	print("Creating datasets...")
	try:
		train_data, valid_data, test_data = get_datasets(COIN, data_aug_factor)
	except:
		raise
	print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")

	#
	# ------------ MODEL TRAINING -----------
	#
	promising_models = common.parse_training_reports(COIN, model_architecture)
	for model_params in promising_models:
		model_filepath = f"models/promising/{COIN}_{model_architecture}_{model_params['model_num']}_param_tuning.pt"
		model = load_model_by_params(model_filepath, model_params)

		dropout, eta, eta_decay = nn.get_model_parameters()
		reports = [f"Model: {model.get_class_name()}", f"Learning rate: {eta}", f"Learning rate decay: {eta_decay}", f"Chance of dropout: {dropout}", f"Batch size: {BATCH_SIZE}", f"Epochs: {EPOCHS}", f"Coin: {COIN}"]

		start_time = time.time()
		print(f"Model #{model_params['model_num']}")

		fully_train(model, train_data, valid_data, start_time, f"models/promising/{COIN}_{model_architecture}_{model_params['model_num']}_lowest_val_loss.pt")



		#
		# ------------ MODEL TESTING -----------
		#
		# load model with lowest validation loss
		model = load_model(model, f"models/promising/{COIN}_{model_architecture}_{model_params['model_num']}_lowest_val_loss.pt")
		report = "EVALUATE TRAINED MODEL"
		REPORTS.append(report)
		print(report)
		model_acc = evaluate_model(model, test_data)

		# save iff accuracy is higher/lower than threshholds
		if model_acc[0] > model_params["accuracy"] and model_acc[3] < common.INACCURACY_THRESHOLD:
			save_filepath = f"models/best/{COIN}_{model_architecture}_{model_params['model_num']}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
			save_model(model, save_filepath)
			with open(f"reports/{COIN}_best_performers.txt", 'a') as f:
				f.write(save_filepath + '\n')

			save_report(COIN, model_architecture)
	

#
# ------------ CONTROLLER METHODS ---------------
#
def fully_automated_training_pipeline():
	'''
	Pipeline involves three steps:

		1.) Parameter tune: 	find the most promising learning rates, decay rates, and dropout rates for the given architecture
		2.) Continue training:	take all the promising models found in the first step and give them more time to train
		3.) Cleanup:			delete all extraneous files created in the first two phases
	'''
#	neural_net_architecture = ["Pi_0", "Pi_1", "Pi_2", "Pi_3", "Pi_4", "Pi_5", "Pi_6", "Pi_7"]
#	neural_net_architecture = ["PC_0", "PC_1", "PC_2", "PC_3", "PC_4", "PC_5", "PC_6"]
#	neural_net_architecture = ["Laptop_0", "Laptop_1", "Laptop_2", "Laptop_3", "Laptop_4"]
	neural_net_architecture = ["Pi_2"]
	
	for model_architecture in neural_net_architecture:
		parameter_tuner(model_architecture)
		continue_training(model_architecture)
		cleanup()



def transfer_learner():
	'''
	Train new models from models successfully trained on other cryptoasset datasets.
	'''
	# create the datasets
	start_time = time.time()
	data_aug_factor = 16 
	coin = "solana"
	print("Creating datasets...")
	try:
		train_data, valid_data, test_data = get_datasets(coin, data_aug_factor)
	except:
		raise
	print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")
	
	# load previously trained model
	best_models = []
	with open("reports/bitcoin_best_performers.txt", 'r') as f:
		best_models = f.read().splitlines() 

	# perform the transfer learning for each model
	for filename in best_models:
		model_params = common.get_model_params("bitcoin", filename)
		if model_params == None:
			print(f"{filename} Not Found. Continuing on to other models.")
			continue
		model = load_model_by_params(filename, model_params)

		# train on novel data
		dropout, eta, eta_decay = nn.get_model_parameters()
		reports = [f"Model: {model.get_class_name()}", f"Learning rate: {eta}", f"Learning rate decay: {eta_decay}", f"Chance of dropout: {dropout}"]

		start_time = time.time()
		print(f"Architecture: {filename}\nModel: {model_params['model_num']}")

		fully_train(model, train_data, valid_data, start_time, f"models/promising/{coin}_{model_params['architecture']}_{model_params['model_num']}_lowest_val_loss.pt")

		# evaluate
		model = load_model(model, f"models/promising/{coin}_{model_params['architecture']}_{model_params['model_num']}_lowest_val_loss.pt")
		report = "EVALUATE TRAINED MODEL"
		REPORTS.append(report)
		print(report)
		model_acc = evaluate_model(model, test_data)

		# save iff accuracy is higher/lower than threshholds
		if model_acc[0] > common.OUTSTANDING_ACCURACY_THRESHOLD and model_acc[3] < common.INACCURACY_THRESHOLD:
			save_filepath = f"models/best/{coin}_{model_params['architecture']}_{model_params['model_num']}_{int(round(model_acc[0], 2) * 100)}-{int(round(model_acc[3], 2))}_{data_aug_factor}xaug.pt"
			save_model(model, save_filepath)
			with open(f"reports/{coin}_best_performers.txt", 'a') as f:
				f.write(save_filepath + '\n')



def benchmark_models(model_coin, test_coin):
	'''
	Cross compares model performance on other datasets, e.g., bitcoin models on the ethereum dataset.
	'''
	# create the datasets
	start_time = time.time()
	data_aug_factor = 0 
	print("Creating datasets...")

	# separate valid, test
	try:
		_, valid_data, test_data = get_datasets(test_coin, data_aug_factor)
	except:
		raise
	# all data together
	data = load_data(test_coin)
	is_data_clean(data)

	data = generate_dataset(data, limit=data.shape[0], offset=0)

	print(f"Datasets created in {(time.time()-start_time)/60:.1f} mins")
	
	# load previously trained model
	best_models = []
	with open(f"reports/{model_coin}_best_performers_all.txt", 'r') as f:
		best_models = f.read().splitlines() 

	most_reliable_models = []
	for filename in best_models:
		model_params = common.get_model_params(model_coin, filename)
		if model_params == None:
			print(f"{filename} Not Found. Continuing on to other models.")
			continue

		model = load_model_by_params(filename, model_params)

		# evaluate
		model = load_model(model, f"{filename}")
		model_acc_all = evaluate_model(model, data, verbose=False)
		model_acc_valid = evaluate_model(model, valid_data, verbose=False)
		model_acc_test = evaluate_model(model, test_data, verbose=False)

		if (model_acc_all[0] > 0.5) and (model_acc_valid[0] > 0.1) and (model_acc_test[0] > 0.4):
			print(f"{filename}")
			print("ALL DATA")
			print_evaluation_status(model_acc_all)
			print("VALIDATION DATA")
			print_evaluation_status(model_acc_valid)
			print("TEST DATA")
			print_evaluation_status(model_acc_test)
			most_reliable_models.append(filename)
		else:
			print(f"Model {filename} performance did not meet the threshold.")

	print(most_reliable_models)



if __name__ == "__main__":
	try:
		fully_automated_training_pipeline()
#		transfer_learner()
#		benchmark_models(model_coin="all", test_coin="ethereum")
	finally:
		print("Program terminated.")

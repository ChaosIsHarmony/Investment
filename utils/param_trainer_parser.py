'''
Parses the Parameter_Tuning_Reports.txt file so that it can automatically set the parameters in the continue_training method of the data_processor file.
'''
import common

def parse_reports(coin, model_architecture):
	try:
		with open(f"reports/{coin}_Parameter_Tuning_Report_{model_architecture}.txt", 'r') as f:
			reports = f.read()
	except:
		print(f"reports/{coin}_Parameter_Tuning_Report_{model_architecture}.txt not found.")
		raise

	models = []
	while len(reports) > 10:
		model = {}
		# Model
		start_ind = reports.find("MODEL") + 7
		end_ind = reports.find('\n', start_ind)
		model["model_num"] =  int(reports[start_ind : end_ind])

		# Architecture
		start_ind = reports.find("PARAMETERS") + 13
		end_ind = reports.find('\n', start_ind)
		model["architecture"] = reports[start_ind : end_ind]

		# Eta | Decay | Dropout
		start_ind = reports.find("eta") + 5
		end_ind = reports.find('|', start_ind) - 1
		model["eta"] = float(reports[start_ind : end_ind])
		
		start_ind = reports.find("decay") + 7
		end_ind = reports.find('|', start_ind) - 1
		model["decay"] = float(reports[start_ind : end_ind])
		
		start_ind = reports.find("dropout") + 9
		end_ind = reports.find('\n', start_ind)
		model["dropout"] = float(reports[start_ind : end_ind])

		# Perfect Decision
		start_ind = reports.find("Decision") + 10
		end_ind = reports.find('\n', start_ind)
		model["accuracy"] = float(reports[start_ind : end_ind])

		# Signal and Answer Exact Opposite
		start_ind = reports.find("Opposite") + 10
		end_ind = reports.find('\n', start_ind)
		model["inaccuracy"] = float(reports[start_ind : end_ind])

		# Add model if meets accuracy threshhold 
		if model["accuracy"] > common.PROMISING_ACCURACY_THRESHOLD and model["inaccuracy"] < common.INACCURACY_THRESHOLD:
			models.append(model)

		# delete up until next model
		start_ind = reports.find("MODEL", end_ind)
		reports = reports[start_ind:]

	return models



def list_promising_model_details(model_architecture):
	models = parse_reports(model_architecture)

	count = 0
	for model in models:
		count += 1
		print(f"Model num: {model['model_num']}")
		print(f"Model acc: {model['accuracy']}")
		print(f"Model bad: {model['inaccuracy']}")
		print()
	
	print(f"{count} promising models found.")


def get_model_params(coin, filename):
	start_ind = filename.find('_') + 1
	end_ind = filename.find('_', start_ind) + 2
	model_architecture = filename[start_ind:end_ind]

	start_ind = filename.find('_', end_ind) + 1
	end_ind = filename.find('_', start_ind)
	model_num = filename[start_ind:end_ind]

	try:
		models = parse_reports(coin, model_architecture)
		for model in models:
			if model["model_num"] == int(model_num):
				return model
	except:
		return None

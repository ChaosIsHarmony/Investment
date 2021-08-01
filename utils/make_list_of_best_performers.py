import glob

coin = "all"

models = glob.glob(f"models/aggregate/{coin}*")

with open(f"reports/{coin}_best_performers.txt", 'w') as f:
	for model in models:
		f.write(model + '\n')

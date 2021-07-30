import glob

coin = "all"

models = glob.glob(f"models/aggregate/{coin}*")

with open(f"reports/{coin}_best_performers_all.txt", 'w') as f:
	for model in models:
		f.write(model + '\n')

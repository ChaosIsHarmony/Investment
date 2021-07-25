import glob

coin = "bitcoin"

models = glob.glob(f"models/best/{coin}*")

with open(f"reports/{coin}_best_performers_all.txt", 'w') as f:
	for model in models:
		f.write(model + '\n')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DECISIONS = ["BUY 3X", "BUY 2X", "BUY X", "HODL", "SELL Y", "SELL 2Y", "SELL 3Y"]

coin = "algorand"

# Load data
data = pd.read_csv("datasets/clean/{coin}_historical_data_clean.csv")

# Split into training, validation, testing
# 70-15-15 split


# Train Classifier
clf = RandomForestClassifier()


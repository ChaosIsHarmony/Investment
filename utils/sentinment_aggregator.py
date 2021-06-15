import requests
import json
import pandas as pd

#data = requests.get("https://api.alternative.me/fng/?limit=0&date_format=cn").json()["data"]
data = pd.read_csv("fear_and_greed_index.csv")
data = pd.DataFrame(data)
data = data.drop(columns=["Unnamed: 0", "value_classification", "time_until_update"])
#print(data.head())
data.to_csv("datasets/raw/fear_and_greed_index.csv")


import requests
import json
import pandas as pd

data = requests.get("https://api.alternative.me/fng/?limit=0&date_format=cn").json()["data"]
data = pd.DataFrame(data)
data = data.drop(columns=["value_classification", "time_until_update"])
data.to_csv("datasets/raw/fear_and_greed_index.csv", index=False)


'''
Reverses a csv's entries (e.g., if A-Z originally, becomes Z-A)
'''

import pandas as pd

data = pd.read_csv("mk.csv")
data = data.reindex(index=data.index[::-1]).reset_index()
data = data.drop(columns=["index"])
data.to_csv("mk_rev.csv")

import pickle 
import pandas as pd 
df = pd.read_csv("tokenized_startups.tsv", sep="\t").fillna("")
lk = list(set(df["lk_industry"]))
lk = sorted([x for x in lk if x !=""  ])
print(len(lk))
with open("industries.pkl", "wb") as outile:
	pickle.dump(lk, outile)
import pickle 
import pandas as pd 


with open("vec_rep.pkl", "rb") as infile:
	vecs = pickle.load(infile)

ids = list(pd.read_csv("tokenized_startups.tsv", sep="\t").fillna("")["id"])

assert len(ids) == len(vecs)

print(len(ids))
print(len(set(ids)))

id_to_vec = {} 
for i in range(len(ids)):
	id_to_vec[ids[i]] = vecs[i]


with open("id_to_vec.pkl", "wb") as outfile:
	pickle.dump(id_to_vec, outfile)
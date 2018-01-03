import pandas as pd 
import numpy as np 
import pickle


def main():
	with open("industries.pkl", "rb") as infile:
		industries = pickle.load(infile)

	tokenized_startups = pd.read_csv("tokenized_startups.tsv", sep="\t").fillna("")
	init_len = len(tokenized_startups)
	tokenized_startups = tokenized_startups[tokenized_startups["lk_industry"] != ""]
	tokenized_startups = tokenized_startups[tokenized_startups["lk_industry"] != "Internet"]
	tokenized_startups = tokenized_startups[tokenized_startups["lk_industry"] != "Computer Software"]
	assert init_len != len(tokenized_startups)


	train, validate, test = np.split(tokenized_startups.sample(frac=1), [int(.6*len(tokenized_startups)), int(.8*len(tokenized_startups))])

	with open("training.pkl", "wb") as outfile:
		pickle.dump(vectorize(train, industries), outfile)


	with open("validation.pkl", "wb") as outfile:
		pickle.dump(vectorize(validate, industries), outfile)


	with open("testing.pkl", "wb") as outfile:
		pickle.dump(vectorize(test, industries), outfile)

def vectorize(df, industries):
	descriptions = list(df["full_desc"])
	descriptions = [pad_sequence(description.split()) for description in descriptions]
	labels = [industries.index(label) for label in list(df["lk_industry"])] 
	desc_label_tup = [] 
	for i in range(len(descriptions)):
		desc_label_tup.append((descriptions[i], labels[i]))
	return desc_label_tup			




def pad_sequence(seq):
	if len(seq) > 186:
		return seq[:186]
	else:
		return seq+[" "]*(186-len(seq))


main()


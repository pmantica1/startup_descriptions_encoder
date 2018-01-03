import pandas as pd 
import numpy as np 
import pickle


def main():
	with open("industries.pkl", "rb") as infile:
		industries = pickle.load(infile)
	tokenized_startups = pd.read_csv("tokenized_startups.tsv", sep="\t").fillna("")
	with open("data.pkl", "wb") as outfile:
		pickle.dump(vectorize(tokenized_startups), outfile)

def vectorize(df):
	descriptions = list(df["full_desc"])
	descriptions = [pad_sequence(description.split()) for description in descriptions]
	return descriptions



def pad_sequence(seq):
	if len(seq) > 186:
		return seq[:186]
	else:
		return seq+[" "]*(186-len(seq))


main()
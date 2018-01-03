import pandas as pd 
import re 
from nltk.tokenize import word_tokenize
import pickle
from tqdm import tqdm 



def main():
	df = pd.read_csv("tokenized_startups.tsv", delimiter="\t")
	df.fillna('', inplace=True)
	word_freq_dic = {} 

	descriptions = list(df["full_desc"])
	for description in tqdm(descriptions, total=230000):
		for token in description.split():
			if token not in word_freq_dic:
				word_freq_dic[token] = [1, None]
			else:
				word_freq_dic[token] = [word_freq_dic[token][0]+1, None]


	with open("all_tokens.pkl", "wb") as outfile:
		pickle.dump(word_freq_dic, outfile)







main()

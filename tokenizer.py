import pandas as pd 
import re 
from nltk.tokenize import word_tokenize
from nltk.tokenize import StanfordTokenizer
import os 
from tqdm import tqdm

jar = "/home/pedro/stanford-postagger-full-2017-06-09/stanford-postagger.jar"
os.environ['STANFORD_POSTAGGER'] = jar 


#model = "/home/pedro/stanford-postagger-full-2017-06-09/models/english-caseless-left3words-distsim.tagger"

#pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

default_tokenizer = StanfordTokenizer()
def main():
	df = pd.read_csv("startups.tsv", delimiter="\t")
	df.fillna('', inplace=True)

	high_concept = df["high_concept"].apply(add_point)
	product_desc = df["product_desc"]

	df["full_desc"] = (high_concept+product_desc)
	df["full_desc"] = df["full_desc"].apply(normalize_description)
	df["full_desc"] = df["full_desc"].apply(tokenize_description)

	#df["full_desc"] = [tokenize_description(desc) for desc in tqdm(list(df["product_desc"]), total=230000)]

	df.to_csv("tokenized_startups.tsv", sep="\t", index=False)
	print(len(df))

def normalize_description(description):
	norm = re.sub(r"http\S+", "", description)
	return norm 



def tokenize_description(description):
	return " ".join(word_tokenize(description)).lower()


def add_point(string):
	if len(string) == 0:
		return string
	elif string[-1]==".":
		return string+" "
	else:
		return string+" . "

main()

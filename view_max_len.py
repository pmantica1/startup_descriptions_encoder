import pandas as pd 

df = pd.read_csv("tokenized_startups.tsv", sep="\t")


token_desc = list(df["full_desc"])
sentence_length = []  
for sentence in token_desc:
	length = len(str(sentence).split())
	sentence_length.append(length)

sentence_length.sort()

print(sentence_length[int(len(sentence_length)-1)])
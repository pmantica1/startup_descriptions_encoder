import pickle 
from tqdm import tqdm
import pprint 
import numpy as np 
import csv 
import sys

csv.field_size_limit(sys.maxsize)

with open("all_tokens.pkl", "rb") as infile:
	tokens = pickle.load(infile)
total = sum([count for count,state in tokens.values()])

count = 0
with open("glove.42B.300d.txt") as infile:
	for line in tqdm(infile, total=1900000):
		real_line = line.strip("\n").split()
		word = real_line[0]
		if word in tokens:
			count += tokens[word][0]
			vector = np.array(real_line[1:])
			assert len(vector) == 300
			tokens[word][1] = np.array(vector)

words = list(tokens.keys())
for word in words:
	if tokens[word][1] is None:
		del tokens[word]
	else:
		tokens[word] = tokens[word][1]
print(count/total)






#pprint.pprint(found_tokens)


with open("found_tokens.pkl", "wb") as outfile:
	pickle.dump(tokens, outfile)
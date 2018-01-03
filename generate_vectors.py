import pandas as pd 
import numpy as np 
import pickle

with open("training.pkl", "rb") as infile:
	training = pickle.load(infile)

with open("found_tokens.pkl", "rb") as infile:
	all_tokens = pickle.load(infile)


def training_vectors(batch_size):
	count = 0
	while(True):
		desc, label = training[count%len(training)]
		vector  generate_vector(desc), label
		count += 1		
	    assert X.shape == (nb_timesteps, word_embedding_dims)
        assert Y.shape == (nb_industries)
        yield X, Y

def generate_vector(desc):
	arrays = [] 
	for token in desc:
		val = all_tokens.get(token, np.zeros(300))
		arrays.append(val)
	a = np.array(arrays)
	return a



from keras.models import load_model
from keras.models import Model 
import numpy as np
import pickle 

with open("data.pkl", "rb") as infile:
	data = pickle.load(infile) #same as training 



with open("found_tokens.pkl", "rb") as infile:
	all_tokens = pickle.load(infile) #dictionary of token to embedding 

nb_timesteps = 186 # 95% percentile of description length 
word_embedding_dims = 300 # according to below message
nb_industries = 107 # idk. 100 something?
batch_size = 1



def input_vectors(source):
	"""
	Feature_vector and label vector generator. The source 
	can be either training, or validation. 
	"""
	count = 0
	while(True):
		X = []
 
		for i in range(batch_size):
			desc = source[count%len(source)]
			X.append(generate_vector(desc))

		X=np.array(X)


		assert X.shape == (batch_size, nb_timesteps, word_embedding_dims)
		yield X


def generate_vector(desc):
	'''
	Generates the feature matrix given a list of tokens
	'''
	arrays = [] 
	for token in desc:
		val = all_tokens.get(token, np.zeros(300))
		arrays.append(val)
	a = np.array(arrays)
	return a


model = load_model('my_model.h5')
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(index=1).output)


intermediate_output = intermediate_layer_model.predict_generator(input_vectors(data), verbose=1, steps=int(len(data)/batch_size))
with open("vec_rep.pkl", "wb") as outfile:
	pickle.dump(intermediate_output, outfile)



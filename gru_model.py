from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import RMSprop
from keras.layers import GRU, LSTM
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import pickle
import math 
import h5py


with open("training.pkl", "rb") as infile:
	training = pickle.load(infile) #tuple of a list of tokens (str)  and labels (ints)
	#Short descriptions are paded with white space tokens at the end 
	#that will correspond to 0 vectors since white space is not vectorized


with open("validation.pkl", "rb") as infile:
	validation = pickle.load(infile)[:10000] #same as training 

with open("testing.pkl", "rb") as infile:
	test = pickle.load(infile) #same as training 



with open("found_tokens.pkl", "rb") as infile:
	all_tokens = pickle.load(infile) #dictionary of token to embedding 

nb_timesteps = 186 # 95% percentile of description length 
word_embedding_dims = 300 # according to below message
nb_industries = 107 # idk. 100 something?
batch_size = 32



def input_vectors(source):
	"""
	Feature_vector and label vector generator. The source 
	can be either training, or validation. 
	"""
	count = 0
	while(True):
		#X is a batch of size 1 of a feature matrix 
		#The input does not accept 2-D feature vectors 
		#that's why I made this transformation. 
		X = []
		#Y is a batch of size of a label vector 
		Y = []   
		for i in range(batch_size):
			desc, label = source[count%len(source)]
			X.append(generate_vector(desc))
			label_vector = [0]*nb_industries
			label_vector[label] = 1
			Y.append(np.array(label_vector))
			count += 1
		X=np.array(X)
		Y=np.array(Y)


		assert X.shape == (batch_size, nb_timesteps, word_embedding_dims)
		assert Y.shape == (batch_size, nb_industries)
		yield X, Y	


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


def get_label_dict(): 
	'''
	Get's a frequency dictionary for the labels
	'''
	freq_dic = {}
	for desc, label in training:
		freq_dic[label] = freq_dic.get(label, 0)+1
	return freq_dic



def create_class_weight(labels_dict,mu=1):
	'''
	Produces class weights based on inverse log frequency
	'''
	total = sum(list(labels_dict.values()))
	keys = labels_dict.keys()
	class_weight = dict()
	for key in keys:
	    score = math.log(total/float(labels_dict[key]))
	    assert score > 0
	    class_weight[key] = score 

	#most_freq_label, max_freq = get_max(labels_dict)
	#class_weight[most_freq_label] = 0.000000001
	#del labels_dict[most_freq_label]

	#second_most_freq_label, second_max_freq = get_max(labels_dict)
	#class_weight[second_most_freq_label] = 0.000000001
	#del labels_dict[second_most_freq_label]

	#print(total/(total-max_freq-second_max_freq))

	return class_weight

def get_max(label_dict):
	max_freq = 0
	most_freq_label = None 
	for label, freq in label_dict.items():
		if freq > max_freq:
			max_freq = freq 
			most_freq_label = label
	return most_freq_label, max_freq 




gru_class_weights = create_class_weight(get_label_dict())


x_in = Input(shape=(nb_timesteps, word_embedding_dims))
hidden = GRU(64, activation='relu')(x_in)
y_out = Dense(nb_industries, activation="softmax")(hidden)
	
model = Model(inputs=x_in, outputs=y_out) 
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(input_vectors(training), validation_data=input_vectors(validation), validation_steps = int(len(validation)/batch_size), 
					 verbose=1, steps_per_epoch=int(len(training)/batch_size), class_weight=gru_class_weights, epochs=8)

print(model.evaluate_generator(input_vectors(test), steps=int(len(test)/batch_size)))
labels = model.predict_generator(input_vectors(test), verbose= 1, steps=int(len(test)/batch_size))



with open("predicted_labels.pkl", "wb") as outfile:
	pickle.dump(labels, outfile)

model.save('my_model.h5')
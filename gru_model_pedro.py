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
	training = pickle.load(infile)[:1000] #tuple of a list of tokens (str)  and labels (ints)
	#Short descriptions are paded with white space tokens at the end 
	#that will correspond to 0 vectors since white space is not vectorized


with open("validation.pkl", "rb") as infile:
	validation = pickle.load(infile)[:1000] #same as training 

with open("found_tokens.pkl", "rb") as infile:
	all_tokens = pickle.load(infile) #dictionary of token to embedding 

nb_timesteps = 186 # 95% percentile of description length 
word_embedding_dims = 300 # according to below message
nb_industries = 107 # idk. 100 something?




def input_vectors(source):
	"""
	Feature_vector and label vector generator. The source 
	can be either training, or validation. 
	"""
	count = 0
	while(True):
		desc, label = source[count%len(source)]

		#X is a batch of size 1 of a feature matrix 
		#The input does not accept 2-D feature vectors 
		#that's why I made this transformation.   
		X = np.array([generate_vector(desc)])


		#Y is a batch of size of a label vector 
		Y = [0]*nb_industries
		Y[label] = 1
		Y = np.array([Y])

		assert X.shape == (1, nb_timesteps, word_embedding_dims)
		assert Y.shape == (1, nb_industries)
		yield X, Y
		count += 1	


def generate_vector(desc):
	empty_count = 0 
	valid_count = 0
	arrays = [] 
	for token in desc:
		val = all_tokens.get(token, np.zeros(300))
		arrays.append(val)
		if(token != " "):
			if list(val) == [0]*300:
				empty_count+=1 
			valid_count+= 1 
	#print((valid_count-empty_count)/(empty_count+1))
	a = np.array(arrays)
	return a


def get_label_dict(): 
	freq_dic = {}
	for desc, label in training:
		freq_dic[label] = freq_dic.get(label, 0)+1
	return freq_dic



def create_class_weight(labels_dict,mu=1):
    total = sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        #print(labels_dict[key]/total)
        score = math.log(total/float(labels_dict[key]))
        #print(score)
        assert score > 0
        class_weight[key] = score 
    #class_weight[47] = 0.000000001
    return class_weight

gru_class_weights = create_class_weight(get_label_dict())
optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

x_in = Input(shape=(nb_timesteps, word_embedding_dims))
hidden = GRU(64, activation='relu')(x_in)
y_out = Dense(nb_industries, activation="softmax")(hidden)
	
model = Model(inputs=x_in, outputs=y_out) 
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(input_vectors(training), verbose=1, steps_per_epoch=len(training), class_weight=gru_class_weights)
labels = model.predict_generator(input_vectors(validation), steps=len(validation))

with open("predicted_labels.pkl", "wb") as outfile:
	pickle.dump(labels, outfile)

model.save('my_model.h5')
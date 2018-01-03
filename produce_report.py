import pickle
import numpy as np 

#Helper file 
#Given the predicted_labels neatly prints the corresponding industries. 




with open("predicted_labels.pkl", "rb") as infile:
	labels = pickle.load(infile)

labels = [np.argmax(label) for label in labels]
with open("industries.pkl", "rb") as infile:
	industries = pickle.load(infile)

freq_dic = {}
for label in labels:
	freq_dic[industries[label]] = freq_dic.get(industries[label], 0)+1

industry_freq_tuples = [] 
for industry, freq in freq_dic.items():
	industry_freq_tuples.append([industry, freq])

industry_freq_tuples = sorted(industry_freq_tuples, key=lambda x: x[1], reverse=True)

for industry, freq in industry_freq_tuples:
	print(industry+":", freq)
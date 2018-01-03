import pickle 
f = open("industries.pkl", "rb")
ind = pickle.load(f)
ind[ind.index("Business Supplies and Equipment_Check")] = "Business Supplies and Equipment"


f.close()

f = open("industries.pkl", "wb")
pickle.dump(ind, f)
f.close()
from copy import deepcopy
import numpy as np
import random

class Dataset():
	def __init__(self, df, desc_field, label_field):
		
		self.df = deepcopy(df)
		self.desc_field = desc_field
		self.label_field = label_field

	
	def getData(self):
		
		data = self.df

		length_count = data.modlAMPLength.value_counts()
		
		train_idx = []
		
		for k, v in length_count.items():
			if v >= 3:
				idx = data[data.modlAMPLength == k].sample(frac=0.8, random_state=int(10000*random.random())).index
			else:
				idx = data[data.modlAMPLength == k].sample(n=1, random_state=int(10000*random.random())).index
			train_idx.extend(idx)
	
		
		train_set = deepcopy(data[data.index.isin(train_idx)])
		test_set = deepcopy(data[~data.index.isin(train_idx)])
		
		x_train, y_train = self.split(train_set)
		x_test, y_test = self.split(test_set)
		
		return x_train, y_train, x_test, y_test
	
	def split(self, data):
		X = deepcopy(np.asarray(data[self.desc_field], dtype=np.float32))
		
		y = deepcopy(np.asarray(data[self.label_field], dtype=np.float32))
		return X, y

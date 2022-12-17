from copy import deepcopy
import numpy as np

class Dataset():
	def __init__(self, df, desc_field, label_field, random_state=0):
		
		self.df = deepcopy(df)
		self.desc_field = desc_field
		self.label_field = label_field
		self.random_state = random_state
	
	def getData(self):
		
		data = self.df
		train_idx=data.sample(frac=0.8).index
		

		train_set = deepcopy(data[data.index.isin(train_idx)])
		test_set = deepcopy(data[~data.index.isin(train_idx)])
		
		x_train, y_train = self.split(train_set)
		x_test, y_test = self.split(test_set)
		
		return x_train, y_train, x_test, y_test
	
	def split(self, data):
		X = deepcopy(np.asarray(data[self.desc_field], dtype=np.float32))
		
		y = deepcopy(np.asarray(data[self.label_field], dtype=np.float32))
		return X, y

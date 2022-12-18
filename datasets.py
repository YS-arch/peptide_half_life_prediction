from copy import deepcopy
import numpy as np
import random
from sklearn import preprocessing

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
			if v >= 2:
				idx = data[data.modlAMPLength == k].sample(frac=0.8, random_state=int(10000*random.random())).index
			else:
				idx = data[data.modlAMPLength == k].sample(n=1, random_state=int(10000*random.random())).index
			train_idx.extend(idx)
	
		
		train_set = deepcopy(data[data.index.isin(train_idx)])
		test_set = deepcopy(data[~data.index.isin(train_idx)])
		
		x_train, y_train = self.split(train_set)
		x_test, y_test = self.split(test_set)
		
		MinMaxScaler = preprocessing.MinMaxScaler()

		x_train = MinMaxScaler.fit_transform(x_train)
		x_test = MinMaxScaler.transform(x_test)
		
		StandardScaler = preprocessing.StandardScaler()
		
		x_train = StandardScaler.fit_transform(x_train)
		x_test = StandardScaler.transform(x_test)
		

		y_train = y_train.ravel()
		y_test = y_test.ravel()
		
		return x_train, y_train, x_test, y_test
	
	def split(self, data):
		X = deepcopy(np.asarray(data[self.desc_field], dtype=np.float32))
		
		y = deepcopy(np.asarray(data[self.label_field], dtype=np.float32))
		return X, y

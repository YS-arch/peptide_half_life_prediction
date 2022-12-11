from copy import deepcopy
import numpy as np
import pandas as pd


class Dataset():
	def __init__(self, df, desc_field, label_field, cut_len, max_len, min_len, random_state=0):

		fitler1 = df[df.modlAMPLength <= max_len]
		fitler2 = fitler1[fitler1.modlAMPLength >= min_len]

		self.df = deepcopy(fitler2)
		self.max_len = max_len
		self.min_len = min_len
		self.cut_len = cut_len
		self.desc_field = desc_field
		self.label_field = label_field
		self.random_state = random_state

	def getData(self):

		data = self.df
		length_count = data.modlAMPLength.value_counts()
		print("length_count:{}".format(length_count))
		train_idx = []
		for k, v in length_count.items():
			if v >= self.cut_len:
				idx = data[data.modlAMPLength == k].sample(frac=0.8, random_state=self.random_state).index
			else:
				idx = data[data.modlAMPLength == k].sample(n=1, random_state=self.random_state).index
			train_idx.extend(idx)

		train_set = deepcopy(data[data.index.isin(train_idx)])
		test_set = deepcopy(data[~data.index.isin(train_idx)])
        
        #此处必须将测试集和训练集分别进行数据处理,不能先预处理再分割数据集
		x_train, y_train = self.split(train_set)
		x_test, y_test = self.split(test_set)

		return x_train, y_train, x_test, y_test

	def split(self, data):
		X = deepcopy(np.asarray(data[self.desc_field],dtype=np.float32))
		
		y = deepcopy(np.asarray(data[self.label_field],dtype=np.float32))
		return X, y

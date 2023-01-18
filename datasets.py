from copy import deepcopy
import numpy as np
import random
from sklearn import preprocessing
from sklearn.utils import shuffle


class Dataset():
	
	def __init__(self, df, desc_field, label_field,transform_fn):
		self.label_field = label_field
		label_data = deepcopy(transform_fn(df[label_field]))
		# 删除方差接近0的特征
		self.desc_field = [col for col in desc_field if df[col].var() > 0.001]
		# 没有做相关系数>0.95的筛选
		df = deepcopy(df[self.desc_field])
		df[label_field] = label_data
		self.df = deepcopy(df)
		
	
	def get_spxy_data(self,test_size):
		data = deepcopy(shuffle(self.df))
		
		x = np.asarray(deepcopy(data[self.desc_field]))
		y = np.asarray(deepcopy(data[self.label_field]))
		
		
		x_train, x_test, y_train, y_test = self.spxy(x, y,test_size)

		MinMaxScaler = preprocessing.MinMaxScaler()
		x_train = MinMaxScaler.fit_transform(x_train)
		x_test = MinMaxScaler.transform(x_test)
		
		StandardScaler = preprocessing.StandardScaler()
		x_train = StandardScaler.fit_transform(x_train)
		x_test = StandardScaler.transform(x_test)
		
		y_train = np.squeeze(y_train)
		y_test = np.squeeze(y_test)
		
		return x_train, y_train, x_test, y_test
	
	def spxy(self, x, y, test_size=0.2):
		"""
		:param x: shape (n_samples, n_features)
		:param y: shape (n_sample, )
		:param test_size: the ratio of test_size
		:return: spec_train :(n_samples, n_features)
				 spec_test: (n_samples, n_features)
				 target_train: (n_sample, )
				 target_test: (n_sample, )
		"""
		x_backup = x
		y_backup = y
		M = x.shape[0]
		N = round((1 - test_size) * M)
		samples = np.arange(M)
		
		y = (y - np.mean(y)) / np.std(y)
		D = np.zeros((M, M))
		Dy = np.zeros((M, M))
		
		for i in range(M - 1):
			xa = x[i, :]
			ya = y[i]
			for j in range((i + 1), M):
				xb = x[j, :]
				yb = y[j]
				D[i, j] = np.linalg.norm(xa - xb)
				Dy[i, j] = np.linalg.norm(ya - yb)
		
		Dmax = np.max(D)
		Dymax = np.max(Dy)
		D = D / Dmax + Dy / Dymax
		
		maxD = D.max(axis=0)
		index_row = D.argmax(axis=0)
		index_column = maxD.argmax()
		
		m = np.zeros(N)
		m[0] = index_row[index_column]
		m[1] = index_column
		m = m.astype(int)
		
		dminmax = np.zeros(N)
		dminmax[1] = D[m[0], m[1]]
		
		for i in range(2, N):
			pool = np.delete(samples, m[:i])
			dmin = np.zeros(M - i)
			for j in range(M - i):
				indexa = pool[j]
				d = np.zeros(i)
				for k in range(i):
					indexb = m[k]
					if indexa < indexb:
						d[k] = D[indexa, indexb]
					else:
						d[k] = D[indexb, indexa]
				dmin[j] = np.min(d)
			dminmax[i] = np.max(dmin)
			index = np.argmax(dmin)
			m[i] = pool[index]
		
		m_complement = np.delete(np.arange(x.shape[0]), m)
		
		spec_train = x[m, :]
		target_train = y_backup[m]
		spec_test = x[m_complement, :]
		target_test = y_backup[m_complement]
		
		return spec_train, spec_test, target_train, target_test
	
	def split(self, data):
		X = deepcopy(np.asarray(data[self.desc_field], dtype=np.float32))
		y = deepcopy(np.asarray(data[self.label_field], dtype=np.float32))
		return X, y
	
	def getData(self):
		
		data = self.df
		
		length_count = data.modlAMPLength.value_counts()
		
		train_idx = []
		
		for k, v in length_count.items():
			if v >= 3:
				idx = data[data.modlAMPLength == k].sample(frac=0.8, random_state=int(10000 * random.random())).index
			else:
				idx = data[data.modlAMPLength == k].sample(n=1, random_state=int(10000 * random.random())).index
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
		
		y_train = np.squeeze(y_train)
		y_test = np.squeeze(y_test)
		
		return x_train, y_train, x_test, y_test
	

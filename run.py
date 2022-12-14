from scipy.stats import pearsonr
import numpy as np
from utils import *
from datasets import Dataset
from descriptors import getMoleculeDesc, getMoleculeAndPeptideDesc, getAllDesc
from sklearn import linear_model, neighbors, svm, ensemble, gaussian_process
from sklearn import preprocessing, metrics
import xgboost
import joblib

"""
WARNING: the onlyHeavy argument to mol.GetNumAtoms() has been deprecated.
Please use the onlyExplicit argument instead or mol.GetNumHeavyAtoms() if you want the heavy atom count.
"""


def myMetrics(y_true, y_pred):
	y_true = np.squeeze(np.asarray(y_true))
	y_pred = np.squeeze(np.asarray(y_pred))
	print(y_true, y_pred)
	mae = metrics.mean_absolute_error(y_true, y_pred)
	
	mse = metrics.mean_squared_error(y_true, y_pred, squared=False)
	
	rmse = metrics.mean_squared_error(y_true, y_pred, squared=True)
	
	r2 = metrics.r2_score(y_true, y_pred)
	
	r = pearsonr(y_true, y_pred)[0]
	
	return r, r2, mae, mse, rmse


def readFileAndCal(path, seqField='SEQUENCE', smiField='SMILES', labelField='T', mode='getMoleculeDesc', before_num=50,
				   before_open=False):
	# 计算前before_num个数据
	data = pd.read_csv(path)
	desc = {}
	
	if mode == 'getAllDesc':
		if before_open:
			desc = getAllDesc(seqToUpper(list(data[seqField]))[:before_num], list(data[smiField])[:before_num])
		else:
			desc = getAllDesc(seqToUpper(list(data[seqField])), list(data[smiField]))
	elif mode == 'getMoleculeAndPeptideDesc':
		if before_open:
			desc = getMoleculeAndPeptideDesc(seqToUpper(list(data[seqField]))[:before_num],
											 list(data[smiField])[:before_num])
		else:
			desc = getMoleculeAndPeptideDesc(seqToUpper(list(data[seqField])), list(data[smiField]))
	elif mode == 'getMoleculeDesc':
		if before_open:
			desc = getMoleculeDesc(seqToUpper(list(data[seqField]))[:before_num], list(data[smiField])[:before_num])
		else:
			desc = getMoleculeDesc(seqToUpper(list(data[seqField])), list(data[smiField]))
	
	df = pd.DataFrame.from_dict(desc)
	
	df[seqField] = data[seqField]
	df[smiField] = data[smiField]
	
	df['label'] = data[labelField]
	
	return df


descs = ['AllDesc', 'MoleculeDesc', 'MoleculeAndPeptideDesc']

tasks = ['Human_blood_nature', 'Human_blood_modification', 'mouse_blood_modification', 'mouse_blood_nature',
		 'mouse_crude_intestinal_modification']

models = {linear_model.LinearRegression: 'LR', linear_model.Perceptron: "Perception",
		  linear_model.Ridge: 'Ridge', linear_model.Lasso: 'Lasso',
		  linear_model.ElasticNet: 'ElasticNet',
		  neighbors.KNeighborsRegressor: 'KNN', svm.SVR: 'SVR_linear',
		  ensemble.RandomForestRegressor: 'RF', ensemble.GradientBoostingRegressor: 'GradientBoostingRegressor',
		  gaussian_process.GaussianProcessRegressor: 'GaussianProcessRegressor',
		  xgboost.XGBRegressor: 'XGBRegressor'}

for task in tasks:
	for desc in descs:
		df = readFileAndCal(path='./data/' + task + '.csv', mode='get' + desc, before_num=50, before_open=False)
		print("{}_{} done!,shape={}".format(task, desc, df.shape))
		df.to_csv("./features/{}_{}.csv".format(task, desc))
		# print(df.columns[:-3])
		data = Dataset(df, list(df.columns[:-3]), ['label'])
		x_train, y_train, x_test, y_test = data.getData()
		print("x_train.shape={},y_train.shape={},x_test.shape={},y_test.shape={}".format(x_train.shape, y_train.shape,
																						 x_test.shape, y_test.shape))
		scaler = preprocessing.StandardScaler()
		
		x_train = scaler.fit_transform(x_train)
		x_test = scaler.transform(x_test)
		
		y_train = y_train.ravel()
		y_test = y_test.ravel()
		
		met = pd.DataFrame(columns=['r', 'r2', 'mae', 'mse', 'rmse'])
		
		for model_class in models.keys():
			model = model_class()
			model.fit(x_train, y_train)
			y_predicted = model.predict(x_test)
			r, r2, mae, mse, rmse = myMetrics(y_true=y_test, y_pred=y_predicted)
			print("r:{}, r2:{}, mae:{}, mse:{}, rmse:{}".format(r, r2, mae, mse, rmse))
			row = {'r': r, 'r2': r2, 'mae': mae, 'mse': mse, 'rmse': mse}
			met = met.append(row, ignore_index=True)
			joblib.dump(model, "./model/{}_{}_{}.pkl".format(task, desc, models[model_class]), compress=3)
			print("{}_{}_{} done!".format(task, desc, models[model_class]))
		print("=" * 20)
		print(met)
		print("=" * 20)
		met.to_csv("./metrics/{}_{}.csv".format(task, desc))

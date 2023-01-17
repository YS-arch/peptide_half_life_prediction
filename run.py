from utils import *
from datasets import Dataset
import pandas as pd
from myModel import *
import numpy as np
import joblib

descs = [
	'MoleculeDesc',
	'MoleculeAndPeptideDesc',
	'AllDesc',
]

tasks = [
	# 'mouse_blood_modification',
	# 'mouse_blood_nature',
	# 'mouse_crude_intestinal_modification',
	'Human_blood_nature',
	# 'Human_blood_modification',
]

models = {
	'PLSRegression': getPLSRegression,
	'DecisionTreeRegressor': getDecisionTreeRegressor,
	'GradientBoostingRegressor': getGradientBoostingRegressor,
	'RandomForestRegressor': getRandomForestRegressor,
	'SVR': getSVR,
}

transform = {
	'divide_60': divide_60,
	'identity': identity,
	'log2': np.log2,
	'ln': np.log,
	'log10': np.log10
}

modes = [
	'kfold',
	# 'loo'
]

noDescriptorsField = ['SMILES', 'T', 'SEQUENCE']

result = pd.DataFrame(columns=['desc', 'task', 'models', 'transform', 'modes', 'r', 'r2', 'mae', 'mse', 'rmse'])

for task in tasks:
	for desc in descs:
		df = pd.read_csv("./data/{}_{}.csv".format(task, desc))
		descField = [col for col in df.columns if col not in noDescriptorsField]
		for transform_str, transform_fn in transform.items():
			print('task:{}\ndesc:{}\ntransform_fn:{}\n'.format(task, desc, transform_fn), '=' * 20)
			data = Dataset(df, descField, ['T'], transform_fn=transform_fn)
			# x_train, y_train, x_test, y_test = data.getData()
			x_train, y_train, x_test, y_test = data.get_spxy_data()
			
			print(
				"x_train.shape={},y_train.shape={},x_test.shape={},y_test.shape={}".format(x_train.shape, y_train.shape,
																						   x_test.shape, y_test.shape))
			for model_str, getModel in models.items():
				for mode in modes:
					# print('task:{}\ndesc:{}\ntransform_fn:{}\n'.format(task, desc, transform_fn), '=' * 20)
					model, search = getModel(x_train=x_train, y_train=y_train, mode=mode)
					y_pred = model.predict(x_test)
					r, r2, mae, mse, rmse = myMetrics(y_test, y_pred)
					print(model, mode, r, r2, mae, mse, rmse)
					joblib.dump(model,
								'./model/{}_{}_{}_{}_{}.pkl'.format(desc, task, model_str, transform_str, mode))
					joblib.dump(search,
								'./search/{}_{}_{}_{}_{}.pkl'.format(desc, task, model_str, transform_str, mode))
					# columns=['desc','task','models','transform','modes','r', 'r2', 'mae', 'mse', 'rmse']
					result.loc[len(result)] = [desc, task, model_str, transform_str, mode, r, r2, mae, mse, rmse]

result.to_csv("./metrics/result.csv", index=False)

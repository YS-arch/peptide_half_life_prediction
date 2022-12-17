import pandas as pd
from utils import *
from myModel import *
from datasets import Dataset
from descriptors import getMoleculeDesc, getMoleculeAndPeptideDesc, getAllDesc
from copy import deepcopy
import joblib
from sklearn import preprocessing

"""
WARNING: the onlyHeavy argument to mol.GetNumAtoms() has been deprecated.
Please use the onlyExplicit argument instead or mol.GetNumHeavyAtoms() if you want the heavy atom count.
"""


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


descs = ['AllDesc',
		 #'MoleculeDesc',
		 #'MoleculeAndPeptideDesc'
		 ]

tasks = ['mouse_blood_modification',
		 #'mouse_blood_nature',
		 #'mouse_crude_intestinal_modification',
		 #'Human_blood_nature',
		 #'Human_blood_modification',
		 ]

noDescriptorsField = ['SMILES', 'label', 'SEQUENCE']

for task in tasks:
	for desc in descs:
		# df = readFileAndCal(path='./data/' + task + '.csv', mode='get' + desc, before_num=50, before_open=False)
		# print("{}_{} done!,shape={}".format(task, desc, df.shape))
		# df.to_csv("./features/{}_{}.csv".format(task, desc))
		# print(df.columns[:-3])
		df = pd.read_csv("./features/{}_{}.csv".format(task, desc))
		descField = [col for col in df.columns if col not in noDescriptorsField]
		data = Dataset(df, descField, ['label'])
		x_train, y_train, x_test, y_test = data.getData()
		print("x_train.shape={},y_train.shape={},x_test.shape={},y_test.shape={}".format(x_train.shape, y_train.shape,
																						 x_test.shape, y_test.shape))
		StandardScaler = preprocessing.StandardScaler()
		
		x_train = StandardScaler.fit_transform(x_train)
		x_test = StandardScaler.transform(x_test)
		
		
		MinMaxScaler = preprocessing.MinMaxScaler()
		
		x_train = MinMaxScaler.fit_transform(x_train)
		x_test = MinMaxScaler.transform(x_test)
		
		y_train = y_train.ravel()
		y_test = y_test.ravel()
		
		met = pd.DataFrame(columns=['r', 'r2', 'mae', 'mse', 'rmse'])
		#print(x_train,y_train)
		print("-"*20)
		print("预处理完毕,开始训练")
		
		
		#search, estimator, inner_cv = getElasticNet(x_train, y_train, descField)
		#search, estimator, inner_cv = getSVM(x_train, y_train, descField)
		search, estimator, inner_cv = getGBR(x_train, y_train, descField)
		
		joblib.dump(search, './search/{}_{}.pkl'.format(task, desc), compress=3)
		
		print("网格搜索最优超参数:{}".format(search.best_params_))
		print("网格搜索最优性能:{}".format(search.best_score_))
		
		estimator.set_params(**search.best_params_)
		estimator.fit(x_train, y_train)
		y_train_predicted = estimator.predict(x_train)
		
		mask = np.asarray(estimator.named_steps['selector'].get_support())
		n_selected_features = sum(mask)
		n_features = x_train.shape[1]
		print('特征选择:{}/{}'.format(n_selected_features, n_features))
		featureName = np.asarray(descField)
		print("未被选择的特征:{}".format(featureName[mask == False]))
		
		r, r2, mae, mse, rmse = myMetrics(y_train, y_train_predicted)
		print("训练集评估:R:{},R2:{},MAE:{},MSE:{},RMSE:{}".format(r, r2, mae, mse, rmse))
		row = {'r': r, 'r2': r2, 'mae': mae, 'mse': mse, 'rmse': mse}
		met = met.append(row, ignore_index=True)
		
		y_predicted = estimator.predict(x_test)
		# print("y_predicted={}".format(y_predicted))
		r, r2, mae, mse, rmse = myMetrics(y_test, y_predicted)
		print("测试集评估:R:{},R2:{},MAE:{},MSE:{},RMSE:{}".format(r, r2, mae, mse, rmse))
		row = {'r': r, 'r2': r2, 'mae': mae, 'mse': mse, 'rmse': mse}
		met = met.append(row, ignore_index=True)
		models = []
		
		for i, (train_index, test_index) in enumerate(inner_cv.split(x_train)):
			estimator = deepcopy(estimator)
			estimator.set_params(**search.best_params_)
			estimator.fit(x_train[train_index], y_train[train_index])
			models.append(estimator)
		
		joblib.dump(models, './model/{}_{}.pkl'.format(task, desc), compress=3)
		print("模型文件已保存")
		
		met.to_csv("./metrics/{}_{}.csv".format(task, desc))
		print("评估文件已保存")

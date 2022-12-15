from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import xgboost
from sklearn import linear_model, neighbors, svm, ensemble, gaussian_process

# 从白到绿的渐变色
my_colormap = LinearSegmentedColormap.from_list("", ["white", "green"])
descs = ['AllDesc', 'MoleculeDesc', 'MoleculeAndPeptideDesc']
models = {linear_model.LinearRegression: 'LR',
		  # linear_model.Perceptron: "Perception", #mouse crude ValueError("Unknown label type: %s" % repr(ys))
		  linear_model.Ridge: 'Ridge', linear_model.Lasso: 'Lasso',
		  linear_model.ElasticNet: 'ElasticNet',
		  neighbors.KNeighborsRegressor: 'KNN', svm.SVR: 'SVR_linear',
		  ensemble.RandomForestRegressor: 'RF', ensemble.GradientBoostingRegressor: 'GradientBoostingRegressor',
		  gaussian_process.GaussianProcessRegressor: 'GaussianProcessRegressor',
		  xgboost.XGBRegressor: 'XGBRegressor'}
tasks = ['mouse_blood_nature', 'mouse_crude_intestinal_modification', 'mouse_blood_modification', 'Human_blood_nature',
		 'Human_blood_modification',
		 ]

metric = 'r'

for task in tasks:
	data = pd.DataFrame()
	for desc in descs:
		met = pd.read_csv("./metrics/{}_{}.csv".format(task, desc))[metric]
		data = data.append(met)
	data.columns = ['LR','Ridge','Lasso','EN','KNN','SVR','RF','GBR','GPR','XGBR']
	data.index = ['All','M','MP']
	cmap = sns.heatmap(data, linewidths=0.8, annot=True, cmap=my_colormap)
	
	plt.xlabel("models", size=15)
	plt.ylabel("descripotrs", size=15)
	plt.title("{} of {}".format(metric, task), size=15)
	plt.tight_layout()
	plt.show()

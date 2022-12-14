from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

desc = ['AllDesc', 'MoleculeAndPeptideDesc', 'MoleculeDesc']
models = ['LR', "Perception",
		  'Ridge', 'Lasso',
		  'ElasticNet',
		  'KNN', 'SVR_linear',
		  'RF', 'GradientBoostingRegressor',
		  'GaussianProcessRegressor',
		  'XGBRegressor']

my_colormap = LinearSegmentedColormap.from_list("", ["white", "green"])
rootpath = 'C:/Users/PC/Desktop/metrics/Human_blood_nature_'
dic = {}
for metric in list(('r','r2','mae','mse','rmse')):
	for de in desc:
		da = pd.read_csv(rootpath + de + '.csv')
		dic[de] = da[metric]
	data = pd.DataFrame.from_dict(dic)
	data.index = models
	data.columns=['All','MP','M']
	cmap = sns.heatmap(data, linewidths=0.8, annot=True, cmap=my_colormap)
	plt.xlabel("model", size=15)
	plt.ylabel("descripotr", size=15)
	plt.title("{} of regression".format(metric), size=15)
	plt.tight_layout()
	plt.show()

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plotLength(file):
	data = pd.read_csv(file)
	length_count = data.modlAMPLength.value_counts()
	fig, ax = plt.subplots()
	X = []
	Y = []
	for x, y in length_count.items():
		X.append(x)
		Y.append(y)
	ax.bar(X, Y)
	plt.show()

def create_heatmap(data,y_ticks,x_ticks):
	my_colormap = LinearSegmentedColormap.from_list("", ["white", "green"])
	
	# 绘制热图
	data.columns=y_ticks
	data.index=x_ticks
	cmap = sns.heatmap(data, linewidths=0.8, annot=True, cmap=my_colormap)
	
	plt.xlabel("function", size=15)
	plt.ylabel("number of features", size=15)
	plt.title("$R^2$ of SVR", size=15)
	plt.tight_layout()
	plt.show()


def create_grid_search_plots(grid_search_file,param_name='param_selector__k'):
	search = joblib.load(grid_search_file)
	search_result_df = pd.DataFrame(search.cv_results_)
	print(search_result_df)
	
	param_display_name = param_name[-1]
	
	# find configuration with best test score for each k
	best_score_per_alpha_index = search_result_df.groupby(param_name)['mean_test_score'].idxmax()
	search_result_df = search_result_df.loc[best_score_per_alpha_index, :]
	
	# convert results to long format
	param_names = [param_name]
	train_split_names = [c for c in search_result_df.columns if
						 c.startswith('split') and c.endswith('train_score')]
	test_split_names = [c for c in search_result_df.columns if
						c.startswith('split') and c.endswith('test_score')]
	
	data = []
	for index, row in search_result_df.iterrows():
		param_values = row[param_names].tolist()
		train_scores = row[train_split_names].tolist()
		test_scores = row[test_split_names].tolist()
		#test_scores=[]
		#train_scores=[]
		for train_score in train_scores:
			data.append(param_values + ['train', train_score, row.mean_train_score, index])
		for test_score in test_scores:
			data.append(param_values + ['test', test_score, row.mean_test_score, index])
	# print(data)
	
	plot_data = pd.DataFrame(
		data, columns=[param_display_name, 'split', 'R2', 'mean', 'index'])
	
	plot_data = plot_data.rename(columns={'split': 'Split'})
	
	fig, ax = plt.subplots(figsize=(9, 4))
	sns.lineplot(
		data=plot_data,
		x=param_display_name, y='R2', hue='Split', hue_order=['train', 'test'], ax=ax
	)
	
	x_ticks = sorted(plot_data[param_display_name].unique())
	#x_ticks = x_ticks[::1]
	ax.set_xticks(x_ticks)
	
	x = search.best_params_[param_name.replace('param_', '')]
	y = search.best_score_
	ax.plot(x, y, '*k', markersize=15, zorder=-1, alpha=0.8,
			color=ax.lines[1].get_color())
	
	ax.set_xlim(plot_data[param_display_name].min(), plot_data[param_display_name].max())
	ax.set_xlabel(param_display_name)
	ax.set_ylabel('Model performance (R2)')
	plt.tight_layout()
	plt.show()



search=joblib.load("MoleculeAndPeptideDesc_Human_blood_nature_DecisionTreeRegressor_log2_kfold.pkl")


import xgboost
from sklearn import linear_model, svm, ensemble, gaussian_process, feature_selection, pipeline, model_selection, metrics
import numpy as np
import random


def getGBR(x_train, y_train, descField):
	regr = ensemble.GradientBoostingRegressor()
	selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression)
	
	estimator = pipeline.Pipeline([
		('selector', selector),
		('regr', regr)
	])
	k_grid = list(np.arange(10, len(descField), 10))
	param_grid = {
		'selector__k': k_grid
	}

	inner_cv = model_selection.RepeatedKFold(n_splits=5, n_repeats=5)
	mcc_scorer = metrics.make_scorer(metrics.r2_score, greater_is_better=True)
	
	search = model_selection.GridSearchCV(
		estimator=estimator, param_grid=param_grid,
		cv=inner_cv, scoring=mcc_scorer, n_jobs=-1,
		refit=False, return_train_score=True, verbose=2
	)
	print("-" * 20)
	print("开始进行网格搜索")
	
	search.fit(x_train, y_train)
	
	return search, estimator, inner_cv

def getElasticNet(x_train, y_train, descField):
	regr = linear_model.ElasticNet(max_iter=10000)
	
	selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression)
	
	estimator = pipeline.Pipeline([
		('selector', selector),
		('regr', regr)
	])
	k_grid = list(np.arange(10, len(descField), 10))
	alpha_grid = list(np.arange(1, 50, 5))
	param_grid = {
		'selector__k': k_grid,
		'regr__alpha': alpha_grid
	}
	inner_cv = model_selection.RepeatedKFold(n_splits=5, n_repeats=5)
	mcc_scorer = metrics.make_scorer(metrics.r2_score, greater_is_better=True)
	
	search = model_selection.GridSearchCV(
		estimator=estimator, param_grid=param_grid,
		cv=inner_cv, scoring=mcc_scorer, n_jobs=-1,
		refit=False, return_train_score=True, verbose=2
	)
	print("-" * 20)
	print("开始进行网格搜索")
	
	search.fit(x_train, y_train)
	
	return search, estimator, inner_cv


def getSVM(x_train, y_train, descField):
	regr = svm.SVR()
	
	selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression)
	
	estimator = pipeline.Pipeline([
		('selector', selector),
		('regr', regr)
	])
	
	k_grid = list(np.arange(50, len(descField), 100))
	C_grid = list(np.arange(1, 20, 5))
	param_grid = {
		'selector__k': k_grid,
		'regr__C': C_grid
	}
	inner_cv = model_selection.RepeatedKFold(n_splits=5, n_repeats=5, random_state=int(10000 * random.random()))
	mcc_scorer = metrics.make_scorer(metrics.r2_score, greater_is_better=True)
	
	search = model_selection.GridSearchCV(
		estimator=estimator, param_grid=param_grid,
		cv=inner_cv, scoring=mcc_scorer, n_jobs=-1,
		refit=False, return_train_score=True, verbose=2
	)
	print("-" * 20)
	print("开始进行网格搜索")
	
	search.fit(x_train, y_train)
	
	return search, estimator, inner_cv

from sklearn import feature_selection, cross_decomposition, pipeline, model_selection, metrics, ensemble, svm, tree
import numpy as np


def getSVR(x_train, y_train, mode):
	regr = svm.SVR()
	
	selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression)
	
	estimator = pipeline.Pipeline([
		('selector', selector),
		('regr', regr)
	])
	
	k_grid = np.arange(40, 200, 20)
	
	c_grid = 10. ** np.arange(-2, 3)
	
	param_grid = {
		'selector__k': k_grid,
		'regr__C': c_grid
	}
	if mode == 'kfold':
		inner_cv = model_selection.RepeatedKFold(n_splits=5, n_repeats=5)
	elif mode == 'loo':
		inner_cv = model_selection.LeaveOneOut()
	else:
		raise "MODE PARAMETER ERROR"
	
	score = metrics.make_scorer(metrics.r2_score)
	
	search = model_selection.GridSearchCV(
		estimator=estimator, param_grid=param_grid,
		cv=inner_cv, scoring=score, n_jobs=-1,
		refit=False, return_train_score=True, verbose=2
	)
	
	search.fit(x_train, y_train)
	
	estimator.set_params(**search.best_params_)
	estimator.fit(x_train, y_train)
	
	return estimator, search


def getDecisionTreeRegressor(x_train, y_train, mode):
	regr = tree.DecisionTreeRegressor()
	
	selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression)
	
	estimator = pipeline.Pipeline([
		('selector', selector),
		('regr', regr)
	])
	
	k_grid = np.arange(40, 200, 20)
	
	max_features_grid = ['sqrt', 'log2']
	max_leaf_nodes_grid = [10,100,1000,10000]
	
	
	param_grid = {
		'selector__k': k_grid,
		'regr__max_features': max_features_grid,
		'regr__max_leaf_nodes': max_leaf_nodes_grid
	}
	
	if mode == 'kfold':
		inner_cv = model_selection.RepeatedKFold(n_splits=5, n_repeats=5)
	elif mode == 'loo':
		inner_cv = model_selection.LeaveOneOut()
	else:
		raise "MODE PARAMETER ERROR"
	
	score = metrics.make_scorer(metrics.r2_score)
	
	search = model_selection.GridSearchCV(
		estimator=estimator, param_grid=param_grid,
		cv=inner_cv, scoring=score, n_jobs=-1,
		refit=False, return_train_score=True, verbose=2
	)
	
	search.fit(x_train, y_train)
	
	estimator.set_params(**search.best_params_)
	estimator.fit(x_train, y_train)
	
	return estimator, search


def getGradientBoostingRegressor(x_train, y_train, mode):
	regr = ensemble.GradientBoostingRegressor()
	
	selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression)
	
	estimator = pipeline.Pipeline([
		('selector', selector),
		('regr', regr)
	])
	
	k_grid = np.arange(40, 200, 20)
	
	n_estimators_grid = [10,100,500,1000]
	
	max_features_grid = [None,'sqrt', 'log2']
	
	param_grid = {
		'selector__k': k_grid,
		'regr__n_estimators': n_estimators_grid,
		'regr__max_features':max_features_grid
	}
	if mode == 'kfold':
		inner_cv = model_selection.RepeatedKFold(n_splits=5, n_repeats=5)
	elif mode == 'loo':
		inner_cv = model_selection.LeaveOneOut()
	else:
		raise "MODE PARAMETER ERROR"
	
	score = metrics.make_scorer(metrics.r2_score)
	
	search = model_selection.GridSearchCV(
		estimator=estimator, param_grid=param_grid,
		cv=inner_cv, scoring=score, n_jobs=-1,
		refit=False, return_train_score=True, verbose=2
	)
	
	search.fit(x_train, y_train)
	
	estimator.set_params(**search.best_params_)
	estimator.fit(x_train, y_train)
	
	return estimator, search


def getRandomForestRegressor(x_train, y_train, mode):
	regr = ensemble.RandomForestRegressor()
	
	selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression)
	
	estimator = pipeline.Pipeline([
		('selector', selector),
		('regr', regr)
	])
	
	k_grid = np.arange(40, 200, 20)
	
	n_estimators_grid = [10, 100, 500, 1000]
	
	max_features_grid = [None, 'sqrt', 'log2']
	
	param_grid = {
		'selector__k': k_grid,
		'regr__n_estimators': n_estimators_grid,
		'regr__max_features': max_features_grid
	}
	
	if mode == 'kfold':
		inner_cv = model_selection.RepeatedKFold(n_splits=5, n_repeats=5)
	elif mode == 'loo':
		inner_cv = model_selection.LeaveOneOut()
	else:
		raise "MODE PARAMETER ERROR"
	
	score = metrics.make_scorer(metrics.r2_score)
	
	search = model_selection.GridSearchCV(
		estimator=estimator, param_grid=param_grid,
		cv=inner_cv, scoring=score, n_jobs=-1,
		refit=False, return_train_score=True, verbose=2
	)
	
	search.fit(x_train, y_train)
	
	estimator.set_params(**search.best_params_)
	estimator.fit(x_train, y_train)
	
	return estimator, search


def getPLSRegression(x_train, y_train, mode):
	regr = cross_decomposition.PLSRegression()
	
	selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression)
	
	estimator = pipeline.Pipeline([
		('selector', selector),
		('regr', regr)
	])
	
	k_grid = np.arange(40, 200, 20)
	
	n_components_grid = [2,5,10,20]
	
	param_grid = {
		'selector__k': k_grid,
		'regr__n_components': n_components_grid,
		
	}
	if mode == 'kfold':
		inner_cv = model_selection.RepeatedKFold(n_splits=5, n_repeats=5)
	elif mode == 'loo':
		inner_cv = model_selection.LeaveOneOut()
	else:
		raise "MODE PARAMETER ERROR"
	
	score = metrics.make_scorer(metrics.r2_score)
	
	search = model_selection.GridSearchCV(
		estimator=estimator, param_grid=param_grid,
		cv=inner_cv, scoring=score, n_jobs=-1,
		refit=False, return_train_score=True, verbose=2
	)
	
	search.fit(x_train, y_train)
	
	estimator.set_params(**search.best_params_)
	estimator.fit(x_train, y_train)
	
	return estimator, search

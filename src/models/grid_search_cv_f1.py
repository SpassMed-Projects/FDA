import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys,psutil
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
 
LogisticRegression_param = { 
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-1,3,5),
    'solver'  : ['liblinear','saga']
    }

LinearDiscriminant_param = {
    'solver': ['svd', 'lsqr', 'eigen']
}

DecisionTree_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [8,12,14,16],
    'min_samples_split': [20, 25, 30], # prevent overfitting
    'min_samples_leaf': list(range(2,7,1)), #also used for prevent overfitting
    'n_jobs': -1
}

RandomForest_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [12,14,16,20,22],
    'min_samples_split': [3, 5, 7], # prevent overfitting
    'min_samples_leaf': list(range(2,7,1)), #also used for prevent overfitting
    'n_jobs': -1,
    'n_estimators': [500,1000],
    'min_impurity_decrease': [0.0, 0.01, 0.02, 0.03]
}

XGBoost_param = {
    'learning_rate': [0.1, 0.01],
    'max_depth': [7, 9, 11],
    'min_child_weight': [1,3],
    'subsample': [0.9, 1.0, 1.1],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'n_estimators': [500, 1000],
    'n_jobs': -1
}

AdaBoost_param = {
    'learning_rate': [0.1, 0.01],
    'base_estimator__max_depth': [3, 5, 7],
    'base_estimator__min_samples_leaf': [8,10,12],
    'n_estimators': [500,1000]
}


LGBM_param = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [500, 1000],
    'max_depth': [2, 5, 7],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'subsample': [0.5, 0.6, 0.7],
    'min_child_samples': [7, 10, 13],
    'n_jobs': -1
}

def calculate_weights(y):
    n_samples=len(y)
    unique, counts = np.unique(y, return_counts=True)
    n_samples0 = counts[0]
    n_samples1 = counts[1]
    w0 = n_samples/(2*n_samples0)
    w1 = n_samples/(2*n_samples1)
    return {0:w0, 1:w1}

def LogisticRegression_Grid_CV(X_train,y_train, LogisticRegression_param):
    estimator = LogisticRegression()
    LogisticRegression_param['class_weight'] = [calculate_weights(y_train)]
    gsearch = GridSearchCV(estimator , param_grid = LogisticRegression_param, scoring='f1', cv=5, random_state = 42)
    gsearch.fit(X_train, y_train)
    return gsearch

def LinearDiscriminant_Grid_CV(X_train,y_train, LinearDiscriminant_param):
    estimator = LinearDiscriminantAnalysis()
    LinearDiscriminant_param['class_weight'] = [calculate_weights(y_train)]
    gsearch = GridSearchCV(estimator , param_grid = LinearDiscriminant_param, scoring='f1', cv=5, random_state = 42)
    gsearch.fit(X_train, y_train)
    return gsearch

def DecisionTree_Grid_CV(X_train,y_train, DecisionTree_param):
    estimator = DecisionTreeClassifier()
    DecisionTree_param['class_weight'] = [calculate_weights(y_train)]
    gsearch = GridSearchCV(estimator , param_grid = DecisionTree_param, scoring='f1', cv=5, random_state = 42)
    gsearch.fit(X_train, y_train)
    return gsearch

def RandomForest_Grid_CV(X_train,y_train, RandomForest_param):
    estimator = RandomForestClassifier()
    RandomForest_param['class_weight'] = [calculate_weights(y_train)]
    gsearch = GridSearchCV(estimator, n_estimator = 1000, param_grid = RandomForest_param, scoring='f1', cv=5, random_state = 42)
    gsearch.fit(X_train, y_train)
    return gsearch

def XGBoost_Grid_CV(X_train,y_train, XGBoost_param):
    estimator = xgb.XGBClassifier()
    XGBoost_param['class_weight'] = [calculate_weights(y_train)]
    gsearch = GridSearchCV(estimator , param_grid = XGBoost_param, scoring='f1', cv=5, random_state = 42)
    gsearch.fit(X_train, y_train)
    return gsearch

def AdaBoost_Grid_CV(X_train,y_train, AdaBoost_param):
    base = DecisionTreeClassifier()
    AdaBoost_param['class_weight'] = [calculate_weights(y_train)]
    estimator = AdaBoostClassifier(base_estimator = base)
    gsearch = GridSearchCV(estimator , param_grid = AdaBoost_param, scoring='f1', cv=5, random_state = 42)
    gsearch.fit(X_train, y_train)
    return gsearch

def LGBM_Grid_CV(X_train,y_train, LGBM_param):
    estimator = LGBMClassifier()
    LGBM_param['class_weight'] = [calculate_weights(y_train)]
    gsearch = GridSearchCV(estimator , param_grid = LGBM_param, scoring='f1', cv=5, random_state = 42)
    gsearch.fit(X_train, y_train)
    return gsearch

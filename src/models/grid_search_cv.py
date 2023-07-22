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
 
def print_best_score(gsearch,param_test):
     # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
 
LogisticRegression_param = { 
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
    }

def LogisticRegression_Grid_CV(X_train,y_train, LogisticRegression_param):
    estimator = LogisticRegression()
    gsearch = GridSearchCV(estimator , param_grid = LogisticRegression_param, scoring='roc_auc', cv=5 )
    gsearch.fit(X_train, y_train)
    gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
    print_best_score(gsearch,LogisticRegression_param)

# Import Data
path = '/home/daisy/FDA_Dataset/inpatient_all_final_1.csv'
df1 = pd.read_csv(path).iloc[:,1:]
print()
# df1.drop(columns = ['Veteran flag','Event date','Marital status', 'Marital status encoded',
#                     'State','Ruca category'], inplace=True)
X_admission1 = df1.drop(columns = ['Readmission', 'Died'])
Y_admission1 = df1[['Readmission']]

# Split Train and Test
X_train_ad1, X_test_ad1, y_train_ad1, y_test_ad1 = train_test_split(X_admission1, Y_admission1, test_size=0.20, random_state=42)

LogisticRegression_Grid_CV(X_train_ad1, y_train_ad1, LogisticRegression_param)


def print_best_score(gsearch,param_test):
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
 
LogisticRegression_param = { 
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-1,3,5),
    'solver'  : ['liblinear','saga']
    }

LinearDiscriminan_param = {
    'solver': ['svd', 'lsqr', 'eigen']
}

LinearDiscriminant_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,4,6,8,10,12]
}

def LogisticRegression_Grid_CV(X_train,y_train, LogisticRegression_param):
    estimator = LogisticRegression()
    gsearch = GridSearchCV(estimator , param_grid = LogisticRegression_param, scoring='roc_auc', cv=5)
    gsearch.fit(X_train, y_train)
    print_best_score(gsearch,LogisticRegression_param)

def LinearDiscriminant_Grid_CV(X_train,y_train, LinearDiscriminant_param):
    estimator = LinearDiscriminantAnalysis()
    gsearch = GridSearchCV(estimator , param_grid = LinearDiscriminant_param, scoring='roc_auc', cv=5)
    gsearch.fit(X_train, y_train)
    print_best_score(gsearch,LinearDiscriminant_param)

def DecisionTree_Grid_CV(X_train,y_train, DecisionTree_param):
    estimator = DecisionTreeClassifier()
    gsearch = GridSearchCV(estimator , param_grid = DecisionTree_param, scoring='roc_auc', cv=5)
    gsearch.fit(X_train, y_train)
    print_best_score(gsearch,DecisionTree_param)



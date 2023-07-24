import pandas as pd
import numpy as np
from importlib import reload
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler
import scipy as sp
import copy,os,sys,psutil
import pickle
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import argparse
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from transformation import *    
from grid_search_cv import *
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 
from sklearn import metrics   #Additional scklearn functions

import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier

dict_model = {
    "modelname": "path"

}

    
def prepare_dataset(data):
    # Import Data
    if target == "readmission":
        X = data.drop(columns = ['readmission within 300 days', 'died_within_900days'])
        y = data[['readmission within 300 days']]
    else:
        X = data.drop(columns = ['died_within_900days'])
        y = data[['died_within_900days']]

        # # Split Train and Test (?? 似乎不用)
        # X_train_ad1, X_test_ad1, y_train_ad1, y_test_ad1 = train_test_split(X_admission1, Y_admission1, test_size=0.20, random_state=42)
        # Transform Data
        transform_steps = [("ImputeNumeric", ImputeNumeric()),
                   ('RemoveSkewnessKurtosis', RemoveSkewnessKurtosis()),
                   ('StandardizeStandardScaler', Standardize(RobustScaler()))]
        transform_pipeline = Pipeline(transform_steps)

        X = transform_pipeline.transform(X)

        # Balance the dataset
        sme = SMOTEENN(random_state=42)
        X, y = sme.fit_resample(X, y)
        
        return X,y

def make_prediction(dataset_name,model_name):
    if dataset_name == "quality":
        path = 'quality_dataset_path'
        data = pd.read_csv(path).iloc[:,1:]
        X,y = prepare_dataset(data)
    else:
        path = 'test_dataset_path'
        data = pd.read_csv(path).iloc[:,1:]
        X,y = prepare_dataset(data)
    clf = pickle.load(open(dict_model[model_name]), 'rb')
    predict_label = clf.predict(X)
    predict_contin = [pair[1] for pair in clf.predict_proba(X)]
    return predict_label,predict_contin

def cal



        


        # # load the model from disk
        # loaded_model = pickle.load(open(filename, 'rb'))
        # result = loaded_model.score(X_test, Y_test)
        # print(result)

    elif 

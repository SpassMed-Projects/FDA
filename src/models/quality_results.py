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
from FDA.src.models.statistics_metrics import *
from sklearn.utils.validation import column_or_1d

dict_model = {
    "modelname": "path"
}

dict_target_info = {
    'mortality': [],
    'mortality_cvd':[],
    'readmission': ['/home/daisy/FDA_Dataset/inpatient_all_final_1.csv', 'modelname'],
    'readmission_cvd': ['/home/daisy/FDA_Dataset/inpatient_all_final_1.csv', 'model_name']
}

def prepare_dataset(target):
    # Import Data
    path =  dict_target_info[target][0]
    data = pd.read_csv(path).iloc[:,1:]
    patientId = pd.DataFrame(data['Internalpatientid'])
   
    if target == "readmission":
        X = data.drop(columns = ['Internalpatientid'])
        y = column_or_1d(data[['readmission within 300 days']])
    elif target == "readmission_cvd":
        X = data.drop(columns = ['Internalpatientid'])
        
    elif target == "mortality":
        X = data.drop(columns = ['Internalpatientid'])
        
    else:
        X = data.drop(columns = ['Internalpatientid','died_by_cvd'])
       
    
    # Transform Data
    transform_steps = [("ImputeNumeric", ImputeNumeric()),
                ('RemoveSkewnessKurtosis', RemoveSkewnessKurtosis()),
                ('StandardizeStandardScaler', Standardize(RobustScaler()))]
    transform_pipeline = Pipeline(transform_steps)

    X = transform_pipeline.transform(X)
    
    return X,patientId

def make_prediction(X,target):
    clf = pickle.load(open(dict_target_info[target][1]), 'rb')
    predict_label = clf.predict(X)
    predict_contin = [pair[1] for pair in clf.predict_proba(X)]
    return predict_label, predict_contin

def make_df():
    X,y,patientId= prepare_dataset(target)
    pred_result = patientId
    for target in dict_target_info:
        predict_label, predict_contin = make_prediction(X,y,target)
        pred_result[target + "_label"] = predict_label
        pred_result[target + "_contin"] = predict_contin
  
    pred_result.to_csv('/home/vivi/FDA/reports/test_predict_result.csv')
    
if __name__ == '__main__':
    make_df()

    
    




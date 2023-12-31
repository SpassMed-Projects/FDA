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
from statistics_metrics import *
from sklearn.utils.validation import column_or_1d


dict_target_info = {
    'mortality': ['/home/daisy/FDA_Dataset/final_allcause_mortality_test_1.csv','/home/vivi/FDA/models/XGBoost_mortality_2.sav'],
    'mortality_cvd':['/home/daisy/FDA_Dataset/final_cvd_mortality_test_1.csv', '/home/vivi/FDA/models/XGBoost_mortality_cvd_2.sav'],
    'readmission': ['/home/daisy/FDA_Dataset/inpatient_all_final_test_1.csv', "/home/vivi/FDA/models/RandomForest_readmission_feature_selection.sav"],
    'readmission_cvd': ['/home/daisy/FDA_Dataset/inpatient_CVD_final_test_1.csv', '/home/vivi/FDA/models/LGBM_readmission_cvd_2.sav']
}

def prepare_dataset(target,feature_names,isLGBM=False):
    # Import Data
    path =  dict_target_info[target][0]
    data = pd.read_csv(path).iloc[:,1:]
    
    if target == "readmission":
        X = data.drop(columns = ['Internalpatientid', 'CVD_readmission', 'readmission within 300 days'])
    elif target == "readmission_cvd":
        X = data.drop(columns = ['Internalpatientid'])
        for name in X.columns:
            X = X.rename(columns = {name:name.replace(' ','_')})
       
    elif target == "mortality":
        X = data.drop(columns = ['Internalpatientid'])

    else:
        X = data.drop(columns = ['Internalpatientid'])
    
    # Transform Data
    transform_steps = [("ImputeNumeric", ImputeNumeric()),
                ('RemoveSkewnessKurtosis', RemoveSkewnessKurtosis(feature_names,isLGBM)),
                ('StandardizeStandardScaler', Standardize(RobustScaler()))]
    transform_pipeline = Pipeline(transform_steps)

    X = transform_pipeline.transform(X)

    X = X[feature_names]
    return X

def make_prediction(X,target,clf):
    predict_label = clf.predict(X)
    predict_contin = [pair[1] for pair in clf.predict_proba(X)]
    return predict_label, predict_contin

def get_patientId(target):
    path =  dict_target_info[target][0]
    data = pd.read_csv(path).iloc[:,1:]
    patientId = pd.DataFrame(data['Internalpatientid'])
    return patientId

def make_df():
    pred_result = get_patientId("mortality")
    for target in dict_target_info:
        print(target)
        target_result = get_patientId(target)
        clf = pickle.load(open(dict_target_info[target][1],'rb'))
        if target == 'readmission_cvd':
            X= prepare_dataset(target, clf.feature_name_,True)
        else:
            X= prepare_dataset(target, clf.feature_names_in_)
        predict_label, predict_contin = make_prediction(X,target,clf)
        target_result[target + "_label"] = predict_label
        target_result[target + "_contin"] = predict_contin
        pred_result = pred_result.merge(target_result, how='left')
    pred_result["readmission_mortality_label"] = pred_result["readmission_label"]+pred_result["mortality_label"]
    pred_result["readmission_mortality_contin"] = pred_result["readmission_contin"]+pred_result["mortality_contin"]
    pred_result.fillna(0,inplace=True)
    pred_result.to_csv('/home/vivi/FDA/reports/quality_predict_result.csv')
    
if __name__ == '__main__':
    make_df()
    print("success")

    
    




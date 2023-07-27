import pandas as pd
import numpy as np
from importlib import reload
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
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

from collections import Counter
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
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
    'mortality': ['/home/daisy/FDA_Dataset/final_allcause_mortality_test_1.csv','/home/vivi/FDA/models/RandomForest_mortality.sav'],
    'mortality_cvd':['/home/daisy/FDA_Dataset/final_cvd_mortality_test_1.csv', '/home/vivi/FDA/models/RandomForest_mortality_cvd.sav'],
    'readmission': ['/home/daisy/FDA_Dataset/inpatient_all_final_test_1.csv', "/home/vivi/FDA/models/LinearDiscriminant_readmission.sav"],
    'readmission_cvd': ['/home/daisy/FDA_Dataset/inpatient_CVD_final_test_1.csv', '/home/vivi/FDA/models/DecisionTree_readmission_cvd.sav']
}


def prepare_dataset(target,feature_names):
    # Import Data
    path =  dict_target_info[target][0]
    data = pd.read_csv(path).iloc[:,1:]
    
    if target == "readmission":
        X = data.drop(columns = ['Internalpatientid', 'CVD_readmission', 'readmission within 300 days'])
        y = column_or_1d(data[['readmission within 300 days']])
    elif target == "readmission_cvd":
        X = data.drop(columns = ['Internalpatientid', 'CVD_readmission', 'readmission within 300 days'])
        y = column_or_1d(data[['CVD_readmission']])
    elif target == "mortality":
        X = data.drop(columns = ['Internalpatientid', 'died_within_125days'])
        y = column_or_1d(data[['died_within_125days']])
    else:
        print(target)
        X = data.drop(columns = ['Internalpatientid','died_by_cvd','Age at death'])
        y = column_or_1d(data[['died_by_cvd']])
    
    # Transform Data
    transform_steps = [("ImputeNumeric", ImputeNumeric()),
                ('RemoveSkewnessKurtosis', RemoveSkewnessKurtosis(feature_names)),
                ('StandardizeStandardScaler', Standardize(RobustScaler()))]
    transform_pipeline = Pipeline(transform_steps)

    X = transform_pipeline.transform(X)

    X = X[feature_names]
    return X,y

def get_patientId(target):
    path = dict_target_info[target][0]
    data = pd.read_csv(path).iloc[:,1:]
    patientId = pd.DataFrame(data['Internalpatientid'])
    return patientId

def make_prediction(X,target,clf):
    predict_label = clf.predict(X)
    predict_contin = [pair[1] for pair in clf.predict_proba(X)]
    return predict_label, predict_contin

def calculate_score(y, predict_label):
    scores = [
        get_AUPRC(y, predict_label),
        get_AUROC(y, predict_label),
        get_Accurarcy(y, predict_label),
        get_SensitSpecific(y, predict_label),
        get_Sensitivity(y, predict_label),
        get_Specificity(y, predict_label),
        get_Precision(y, predict_label),
        get_NPC(y, predict_label),
        get_PositiveLR(y, predict_label),
        get_NegativeLR(y, predict_label),
        get_F1score(y, predict_label)
    ]
    return scores

def make_df():
    statistics_metrics = pd.DataFrame(['Area under the precision recall curve (AUPRC)',
                                       'Area under the Receiver Operating Characteristic (AUROC)',
                                       'Overall Accuracy',
                                       'Sum of Sensitivity and Specificity',
                                       'Sensitivity',
                                       'Specificity',
                                       'Precision',
                                       'Negative Predictive Value',
                                       'Positive Likelihood Ratio',
                                       'Negative Likelihood Ratio',
                                       'F1 score'], columns=['statistics_metrics'])
    pred_result = get_patientId("mortality")
    for target in dict_target_info:
        target_result = get_patientId(target)
        clf = pickle.load(open(dict_target_info[target][1],'rb'))
        X, y= prepare_dataset(target, clf.feature_names_in_)
        predict_label, predict_contin = make_prediction(X,target,clf)
        scores = calculate_score(y, predict_label)
        target_result[target + "_label"] = predict_label
        target_result[target + "_contin"] = predict_contin
        statistics_metrics[target] = scores
        print(predict_label)
        pred_result = pred_result.merge(target_result, how='left', on = 'Internalpatientid')
    
    pred_result["readmission_mortality"] = pred_result["readmission_contin"]+pred_result["mortality_contin"]
    pred_result.to_csv('/home/vivi/FDA/reports/test_predict_result.csv')
    statistics_metrics.to_csv('/home/vivi/FDA/reports/test_statistics_metrics.csv')

if __name__ == '__main__':
    make_df()
    print('success')

    
    




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

dict_model = {
    "modelname": "path"

}

dict_target_info = {
    'readmission': ['/home/daisy/FDA_Dataset/inpatient_all_final_1.csv', 'modelname'],
    'readmission_cvd': ['/home/daisy/FDA_Dataset/inpatient_all_final_1.csv', 'model_name']
}
    
def prepare_dataset(target):
    # Import Data
    path =  dict_target_info[target][0]
    data = pd.read_csv(path).iloc[:,1:]
    patientId = pd.DataFrame(data['Internalpatientid'])

    if target == "readmission":
        X = data.drop(columns = ['readmission within 300 days', 'Internalpatientid','died_within_900days'])
        y = data[['readmission within 300 days']]
    else:
        X = data.drop(columns = ['died_within_900days','Internalpatientid'])
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
    
    return X,y,patientId

def make_prediction(X,y,target):
    clf = pickle.load(open(dict_target_info[target][1]), 'rb')
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

def make_df(patientId):
    statistics_metrics = 
    pred_result = patientId
    for target in dict_target_info:
        X,y = prepare_dataset(target)
        predict_label, predict_contin = make_prediction(X,y,target)
        scores = calculate_score(y, predict_label)
        pred_result[target + " label"] = predict_label
        pred_result[target + " continu"] = predict_label
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--isdev", help="run to test pipeline: 1 - True, 0 - False", type=int)

    # Ideally target should be readmission, readmission_cvd, motality, motality_cvd
    parser.add_argument("--model_type", help="select model architecture", type=str)
    parser.add_argument("--target", help="select target", type=str)
    
    args = parser.parse_args()

    print(f"Selected model type is: {args.model_type}")
    print(f"Target: {args.target}")

    X, y = prepare_dataset(args.target)
    predict_label, predict_contin, y = make_prediction(X,y,)
    calculate_score(y, predict_label)

    filename = f"/home/vivi/FDA/models/{args.model_type}_{args.target}.sav"
    pickle.dump(clf, open(filename, 'wb'))

    
    




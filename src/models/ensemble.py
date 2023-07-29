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
import train_model
from test_results import *
from statistics_metrics import *
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
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Ideally target should be readmission, readmission_cvd, motality, motality_cvd
    parser.add_argument("--target", help="select target", type=str)
    
    args = parser.parse_args()


    models = ["DecisionTree", "LinearDiscriminant", "LogisticRegression", "RandomForest"]
    predicted_all_label = pd.DataFrame(columns=models)
    y = []
    for m in models:
        model_name = f"/home/vivi/FDA/models/{m}_{args.target}_2.sav"
        clf = pickle.load(open(model_name,'rb'))

        X, y = prepare_dataset(args.target, clf.feature_names_in_)
        target_result = get_patientId(args.target)

        predict_label, predict_contin = make_prediction(X,args.target,clf)
        predicted_all_label[m] = predict_label
        print("accuracy score for model:", m, "and target:", args.target, get_F1score(y, predict_label))
    
    ensembled_target = predicted_all_label.mode(axis=1).iloc[:,0]
    print(get_F1score(ensembled_target, y))
    
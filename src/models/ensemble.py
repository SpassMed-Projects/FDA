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
import test_results 
from statistics_metrics import *
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 
from sklearn import metrics   #Additional scklearn functions

import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from statistics_metrics import *
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import accuracy_score

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

def ensemble(target):

    model_names = ["DecisionTree", "LinearDiscriminant", "LogisticRegression", "RandomForest", "LGBM", "XGBoost"]

    X_train, y_train = train_model.prepare_dataset(target)
    models = []
    for m in model_names:
        model_name = f"/home/vivi/FDA/models/{m}_{target}_2.sav"
        clf = pickle.load(open(model_name,'rb'))

        models.append((m, clf))

    eclf = VotingClassifier(estimators=models, voting='hard')
    eclf.fit(X_train,y_train)

    X_test, y_test = test_results.prepare_dataset(target, eclf.feature_names_in_)

    # predict_label, predict_contin = test_results.make_prediction(X_test,args.target,eclf)
    predict_label = eclf.predict(X_test)
    return test_results.calculate_score(y_test, predict_label)


if __name__ == '__main__':
    
    targets = ["mortality", "readmission", "readmission_cvd"]

    for target in targets:
        score = ensemble(target)
        statistics_metrics[target] = score
    statistics_metrics.to_csv("/home/vivi/FDA/reports/ensembled_test_statistics_metrics.csv")
    
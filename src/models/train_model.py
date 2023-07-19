import pandas as pd
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import argparse

from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# http://blog.hubwiz.com/2019/09/24/scikit-learn-pipeline-guide/

path = '/home/daisy/FDA_Dataset/inpatient_all_final_1.csv'
df1 = pd.read_csv(path).iloc[:,1:]
df1.drop(columns = ['Veteran flag','Event date','Marital status', 'Marital status encoded',
                    'State','Ruca category'], inplace=True)


path = '/home/daisy/FDA_Dataset/inpatient_all_final_2.csv'
df2 = pd.read_csv(path).iloc[:,1:]
df2.drop(columns = ['Veteran flag','Event date','Marital status', 'Marital status encoded',
                    'State','Ruca category'], inplace=True)

X_admission1 = df1.drop(columns = ['Readmission', 'Died'])
Y_admission1 = df1[['Readmission']]

X_mortality1 = df1.drop(columns = ['Died'])
Y_mortality1 = df1[['Died']]


X_train_ad1, X_test_ad1, y_train_ad1, y_test_ad1 = train_test_split(X_admission1, Y_admission1, test_size=0.20, random_state=42)
X_train_mor1, X_test_mor1, y_train_mor1, y_test_mor1 = train_test_split(X_mortality1, Y_mortality1, test_size=0.20, random_state=42)

missing_cols = df1.columns[df1.isna().any()].tolist()
X_train_ad1[missing_cols] = X_train_ad1[missing_cols].fillna(X_train_ad1[missing_cols].mean())
X_train_mor1[missing_cols] =  X_train_mor1[missing_cols].fillna(X_train_mor1[missing_cols].mean())

X_test_ad1[missing_cols] = X_test_ad1[missing_cols].fillna(X_test_ad1[missing_cols].mean())
X_test_mor1[missing_cols] = X_test_mor1[missing_cols].fillna(X_test_mor1[missing_cols].mean())
# 'Internalpatientid' is not in these colnames
targets = ['Readmission', 'Died']

cat_cols = ['AO', 'CVD', 'Ruca category encoded', 'Ethnicity', 
            'Gender', 'Races', 'Ethnicity_0',
            'Ethnicity_1', 'Ethnicity_2', 'Races_0', 
            'Races_1', 'Races_2', 'Races_3','DOMICILIARY', 
            'MEDICINE', 'NHCU', 'NON-COUNT', 'OTHERS', 'PSYCHIATRY']

numeric_cols = ['num_stays', 'stay_length', 'num_unique_units',
       'num_transfers', 'num_cvd_readmission', 'unique_admitting_specialty', 
       'unique_discharging_specialty','Age 20-40', 'Age 40-60', 'Age 60-80', 'Age 80-100',
       'Age 100-120', 'age_mean', 'age_std', 'age_min', 'age_max', 'stay_min',
       'stay_max', 'stay_mean', 'stay_std', 'freq', 'total_procedure',
       'num_surgery_pro', 'num_immunization', 'Num med per admission mean',
       'Num med per admission min', 'Num med per admission max',
       'Total medications', 'mean age at specailty', 'period mean', 
       'specialty medical count', 'specialty support count',
       'period std','specialty count', 'Age 20-40 hypotension',
       'Age 40-60 hypotension', 'Age 60-80 hypotension',
       'Age 80-100 hypotension', 'Age 100-120 hypotension',
       'Age 20-40 hypertension', 'Age 40-60 hypertension',
       'Age 60-80 hypertension', 'Age 80-100 hypertension',
       'Age 100-120 hypertension', 'Age 20-40 healthy', 'Age 40-60 healthy',
       'Age 60-80 healthy', 'Age 80-100 healthy', 'Age 100-120 healthy',
       'lab_count', 'lab_freq', 'lab_age_mean', 'lab_age_std']

def check_skewness(df):
    statusdf = pd.DataFrame()
    statusdf['numeric_col'] = numeric_cols
    transform = []
    sknewness_before = []
    kurtosis_before = []
    std_before = []
    
    skewness_after = []
    kurtosis_after = []
    std_after = []

    method = []
    for i in numeric_cols:
        if abs(df[i].skew()) > 1.96 and abs(df[i].kurtosis() > 1.96):
            transform.append('Yes')
            sknewness_before.append(df[i].skew())
            kurtosis_before.append(df[i].kurtosis())
            std_before.append(df[i].std())

            skewness_after.append(np.log1p(df[df[i] >= 0][i]).skew())
            kurtosis_after.append(np.log1p(df[df[i] >= 0][i]).kurtosis())
            std_after.append(np.log1p(df[df[i] >= 0][i]).std())

            method.append('log')
        else:
            transform.append('No')
            sknewness_before.append(df[i].skew())
            kurtosis_before.append(df[i].kurtosis())
            std_before.append(df[i].std())

            skewness_after.append(df[i].skew())
            kurtosis_after.append(df[i].kurtosis())
            std_after.append(df[i].std())
            method.append(' ')

    statusdf['transform'] = transform
    statusdf['method'] = method
    statusdf['sknewness_before'] = sknewness_before
    statusdf['skewness_after'] = skewness_after

    statusdf['kurtosis_before'] = kurtosis_before
    statusdf['kurtosis_after'] = kurtosis_after
    
    statusdf['std_before'] = std_before
    statusdf['std_after'] = std_after
    return statusdf

statusdf = check_skewness(df)

def remove_skewness(df,statusdf):
    for i in range(len(statusdf)):
        if statusdf['transform'][i] == 'Yes':
            colname = str(statusdf['numeric_col'][i])
            
            # will lose information here,
            # For np.log() has 'inf', and we will not consider 'inf'
            #df[colname + "_log"] = np.log1p(df[df[colname] >= 0][colname])
            df[colname + "_log"] = np.log1p(df[colname])
    return df


df_log = remove_skewness(df,statusdf)
log_numeric_cols = ['num_stays_log', 'stay_length_log',
       'num_transfers_log', 'num_cvd_readmission_log',
       'unique_admitting_specialty_log', 'Age 20-40_log', 'Age 40-60_log',
       'Age 60-80_log', 'Age 80-100_log', 'Age 100-120_log', 'stay_min_log',
       'stay_max_log', 'stay_mean_log', 'stay_std_log', 'freq_log',
       'total_procedure_log', 'num_surgery_pro_log',
       'Num med per admission mean_log', 'Num med per admission min_log',
       'Num med per admission max_log', 'Total medications_log',
       'period mean_log', 'specialty medical count_log',
       'specialty support count_log', 'period std_log', 'specialty count_log',
       'Age 20-40 hypotension_log', 'Age 40-60 hypotension_log',
       'Age 60-80 hypotension_log', 'Age 80-100 hypotension_log',
       'Age 100-120 hypotension_log', 'Age 20-40 hypertension_log',
       'Age 40-60 hypertension_log', 'Age 60-80 hypertension_log',
       'Age 80-100 hypertension_log', 'Age 100-120 hypertension_log',
       'Age 20-40 healthy_log', 'Age 40-60 healthy_log',
       'Age 60-80 healthy_log', 'Age 80-100 healthy_log',
       'Age 100-120 healthy_log', 'lab_count_log', 'lab_freq_log']
log_cols = ['Internalpatientid'] + log_numeric_cols + cat_cols
df_log = df_log[log_cols]
df_log

from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

#std_scaler = StandardScaler()

def rob_scale_numeric_data(df,cols):
    rob_scaler = RobustScaler()
    for i in cols:
        #new_i =  rob_scaler.fit_transform(df[i].values.reshape(-1,1))
        df[i] = rob_scaler.fit_transform(df[i].values.reshape(-1,1))
        df = df.rename(columns = {i:i+ "_rob_scaled"})
    return df

df_log_norm= rob_scale_numeric_data(df_log,log_numeric_cols)
df_log_norm


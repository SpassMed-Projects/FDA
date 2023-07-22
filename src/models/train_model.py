import pandas as pd
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from importlib import reload

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler

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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from transformation import RemoveSkewnessKurtosis, Standardize


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import Data
path = '/home/daisy/FDA_Dataset/inpatient_all_final_1.csv'
df1 = pd.read_csv(path).iloc[:,1:]
df1.drop(columns = ['Veteran flag','Event date','Marital status', 'Marital status encoded',
                    'State','Ruca category'], inplace=True)
X_admission1 = df1.drop(columns = ['Readmission', 'Died'])
Y_admission1 = df1[['Readmission']]

# Split Train and Test
X_train_ad1, X_test_ad1, y_train_ad1, y_test_ad1 = train_test_split(X_admission1, Y_admission1, test_size=0.20, random_state=42)

# # Transform Data
# targets = ['Readmission', 'Died']

# cat_cols = ['AO', 'CVD', 'Ruca category encoded', 'Ethnicity', 
#             'Gender', 'Races', 'Ethnicity_0',
#             'Ethnicity_1', 'Ethnicity_2', 'Races_0', 
#             'Races_1', 'Races_2', 'Races_3','DOMICILIARY', 
#             'MEDICINE', 'NHCU', 'NON-COUNT', 'OTHERS', 'PSYCHIATRY']

# numeric_cols = ['num_stays', 'stay_length', 'num_unique_units',
#        'num_transfers', 'num_cvd_readmission', 'unique_admitting_specialty', 
#        'unique_discharging_specialty','Age 20-40', 'Age 40-60', 'Age 60-80', 'Age 80-100',
#        'Age 100-120', 'age_mean', 'age_std', 'age_min', 'age_max', 'stay_min',
#        'stay_max', 'stay_mean', 'stay_std', 'freq', 'total_procedure',
#        'num_surgery_pro', 'num_immunization', 'Num med per admission mean',
#        'Num med per admission min', 'Num med per admission max',
#        'Total medications', 'mean age at specailty', 'period mean', 
#        'specialty medical count', 'specialty support count',
#        'period std','specialty count', 'Age 20-40 hypotension',
#        'Age 40-60 hypotension', 'Age 60-80 hypotension',
#        'Age 80-100 hypotension', 'Age 100-120 hypotension',
#        'Age 20-40 hypertension', 'Age 40-60 hypertension',
#        'Age 60-80 hypertension', 'Age 80-100 hypertension',
#        'Age 100-120 hypertension', 'Age 20-40 healthy', 'Age 40-60 healthy',
#        'Age 60-80 healthy', 'Age 80-100 healthy', 'Age 100-120 healthy',
#        'lab_count', 'lab_freq', 'lab_age_mean', 'lab_age_std']

# log_numeric_cols = ['num_stays_log', 'stay_length_log',
#        'num_transfers_log', 'num_cvd_readmission_log',
#        'unique_admitting_specialty_log', 'Age 20-40_log', 'Age 40-60_log',
#        'Age 60-80_log', 'Age 80-100_log', 'Age 100-120_log', 'stay_min_log',
#        'stay_max_log', 'stay_mean_log', 'stay_std_log', 'freq_log',
#        'total_procedure_log', 'num_surgery_pro_log',
#        'Num med per admission mean_log', 'Num med per admission min_log',
#        'Num med per admission max_log', 'Total medications_log',
#        'period mean_log', 'specialty medical count_log',
#        'specialty support count_log', 'period std_log', 'specialty count_log',
#        'Age 20-40 hypotension_log', 'Age 40-60 hypotension_log',
#        'Age 60-80 hypotension_log', 'Age 80-100 hypotension_log',
#        'Age 100-120 hypotension_log', 'Age 20-40 hypertension_log',
#        'Age 40-60 hypertension_log', 'Age 60-80 hypertension_log',
#        'Age 80-100 hypertension_log', 'Age 100-120 hypertension_log',
#        'Age 20-40 healthy_log', 'Age 40-60 healthy_log',
#        'Age 60-80 healthy_log', 'Age 80-100 healthy_log',
#        'Age 100-120 healthy_log', 'lab_count_log', 'lab_freq_log']

transform_steps = [('RemoveSkewnessKurtosis', RemoveSkewnessKurtosis(targets, cat_cols, numeric_cols, log_numeric_cols)),
         ('StandardizeStandardScaler', Standardize(cols, scalar))]
transform_pipeline = Pipeline(transform_steps)

data_prepared = transform_pipeline.fit(X_train_ad1)







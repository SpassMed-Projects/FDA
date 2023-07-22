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

class RemoveSkewnessKurtosis(BaseEstimator, TransformerMixin):
  def __init__(self, targets, cat_cols, numeric_cols, log_numeric_cols):  
    self.targets = targets
    self.cat_cols = cat_cols
    self.numeric_cols = numeric_cols
    self.log_numeric_cols = log_numeric_cols
  
  def check_skewness(self, X):
    statusdf = pd.DataFrame()
    statusdf['numeric_col'] = self.numeric_cols
    transform = []
    sknewness_before = []
    kurtosis_before = []
    std_before = []
    
    skewness_after = []
    kurtosis_after = []
    std_after = []

    method = []
    for i in self.numeric_cols:
        if abs(X[i].skew()) > 1.96 and abs(X[i].kurtosis()) > 1.96:
            transform.append('Yes')
            sknewness_before.append(X[i].skew())
            kurtosis_before.append(X[i].kurtosis())
            std_before.append(X[i].std())

            skewness_after.append(np.log1p(X[X[i] >= 0][i]).skew())
            kurtosis_after.append(np.log1p(X[X[i] >= 0][i]).kurtosis())
            std_after.append(np.log1p(X[X[i] >= 0][i]).std())

            method.append('log')
        else:
            transform.append('No')
            sknewness_before.append(X[i].skew())
            kurtosis_before.append(X[i].kurtosis())
            std_before.append(X[i].std())

            skewness_after.append(X[i].skew())
            kurtosis_after.append(X[i].kurtosis())
            std_after.append(X[i].std())
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
  
  def remove_skewness(self, X):
    statusdf = self.check_skewness(X)
    for i in range(len(statusdf)):
        if statusdf['transform'][i] == 'Yes':
            colname = str(statusdf['numeric_col'][i])
            
            # will lose information here,
            # For np.log() has 'inf', and we will not consider 'inf'
            #df[colname + "_log"] = np.log1p(df[df[colname] >= 0][colname])
            X[colname + "_log"] = np.log1p(X[colname])
    return X

  def extract_log_col(self, X):
    df_log = self.remove_skewness(X)
    log_cols = ['Internalpatientid'] + self.log_numeric_cols + self.cat_cols
    df_log = df_log[log_cols]
    return df_log

  def fit(self, X,  y=None):
    return self
 
  def transform(self, X):
    return self.extract_log_col(X)
  
class Standardize(BaseEstimator, TransformerMixin):
  def __init__(self, cols, scalar):
    self.cols = cols
    self.scalar = scalar

  def rob_scale_numeric_data(self, X, cols):
    for i in cols:
        X[i] = self.scalar.fit_transform(X[i].values.reshape(-1,1))
        X = X.rename(columns = {i:i+ "_rob_scaled"})
    return X

  def fit(self, X,  y=None):
    return self
  
  def transform(self, X):
    return self.rob_scale_numeric_data(X,self.cols)

class ImputeNumeric(BaseEstimator, TransformerMixin):
  def __init__(self):
     pass
  
  def fit(self, X,  y=None):
    return self
  
  def transform(self, X):
    missing_cols = X.columns[X.isna().any()].tolist()
    for colname in missing_cols:
      X[colname].fillna((X[colname].mean()), inplace = True)
    return X
  
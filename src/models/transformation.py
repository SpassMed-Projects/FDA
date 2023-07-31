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
  def __init__(self, feature_name = None):  
    self.targets = ['readmission within 300 days', 'died_within_900days']
    self.cat_cols = ['CVD','Ethnicity', 'Gender', 'Races', 'Ethnicity_0', 'Ethnicity_1', 
            'Ethnicity_2', 'Races_0', 'Races_1', 'Races_2', 'Races_3', 
            'Ruca category encoded']
    self.feature_name = feature_name

  def check_skewness(self, X):
    if self.feature_name is None:
      numeric_cols = list(set(X.columns)- set(self.targets) - set(self.cat_cols))
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
    else:
      numeric_cols = list(set(X.columns)- set(self.targets) - set(self.cat_cols))
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
      feature_name = set(self.feature_name)
      for i in numeric_cols:
          i_log_scaled = i + "_log_rob_scaled"
          if i_log_scaled in feature_name:
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
  

  def extract_log_col(self, X):
    statusdf = self.check_skewness(X)
    for i in range(len(statusdf)):
       if statusdf['transform'][i] == 'Yes':
        colname = str(statusdf['numeric_col'][i])
            
            # will lose information here,
            # For np.log() has 'inf', and we will not consider 'inf'
            #df[colname + "_log"] = np.log1p(df[df[colname] >= 0][colname])
        X[colname+'_log'] = np.log1p(X[colname])
    log_numeric_cols = [x for x in X.columns if '_log' in x]  
    cols_no_transform = list(statusdf[statusdf['transform'] == 'No']['numeric_col'])
    log_cols = log_numeric_cols + self.cat_cols + cols_no_transform 
    X = X[log_cols]
    return X

  def fit(self, X,  y=None):
    return self
 
  def transform(self, X):
    return self.extract_log_col(X)
  
class Standardize(BaseEstimator, TransformerMixin):
  def __init__(self, scalar):
    self.scalar = scalar

  def rob_scale_numeric_data(self, X):
    log_numeric_cols = [x for x in X.columns if '_log' in x]
    for i in log_numeric_cols:
        X[i] = self.scalar.fit_transform(X[i].values.reshape(-1,1))
        X = X.rename(columns = {i:i+ "_rob_scaled"})
    return X

  def fit(self, X,  y=None):
    return self
  
  def transform(self, X):
    return self.rob_scale_numeric_data(X)

class ImputeNumeric(BaseEstimator, TransformerMixin):
  def __init__(self):
     pass
  
  def fit(self, X,  y=None):
    return self
  
  def transform(self, X):
    missing_cols = X.columns[X.isna().any()].tolist()
    for colname in missing_cols:
      print(colname)
      if pd.isna(X[colname]).sum() == len(X[colname]):
         X[colname].fillna((0), inplace = True)
      else:
         X[colname].fillna((X[colname].mean()), inplace = True)
    return X
  
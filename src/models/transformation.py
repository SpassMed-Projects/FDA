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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



# http://blog.hubwiz.com/2019/09/24/scikit-learn-pipeline-guide/

class CustomTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, feature_name, additional_param = "SM"):  
    print('\n...intializing\n')
    self.feature_name = feature_name
    self.additional_param = additional_param
 
  def fit(self, X, y = None):
    print('\nfiting data...\n')
    print(f'\n \U0001f600  {self.additional_param}\n')
    return self
 
  def transform(self, X, y = None):
    print('\n...transforming data \n')
    X_ = X.copy()
    X_[self.feature_name] = np.log(X_[self.feature_name])
    return X
  
print("creating second pipeline...")
pipe2 = Pipeline(steps=[
                       ('experimental_trans', CustomTransformer('bmi')),
                       ('linear_model', LinearRegression())
])
 
print("fiting pipeline 2")
pipe2.fit(X_train, y_train)
preds2 = pipe2.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds2))}\n")

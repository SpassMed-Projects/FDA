''' 
Metrics include:
- Area Under the Precision-Recall Curve (AUPRC)
- Area Under the Reciever Operating Characteristic (AUROC)
- Overall Accurarcy
- Sensitivity + Specificity
https://sinyi-chou.github.io/python-sklearn-precision-recall/
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(401)


def get_AUPRC(y_test, y_score):
    average_precision_score(y_test, y_score)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall, precision)
    print(auc_precision_recall)
    

def get_AUROC():
    pass


def get_Accurarcy():
    pass


def get_SensitSpecific():
    pass


## 不知道有没有用：

# def accuracy(C):
#     ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
#     return np.sum(np.diag(C)) / np.sum(C)

# def recall(C):
#     ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
#     return np.diag(C) / np.sum(C, axis=1)

# def precision(C):
#     ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
#     return np.diag(C) / np.sum(C, axis=0)
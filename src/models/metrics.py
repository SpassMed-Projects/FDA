''' 
Metrics include:
- Area Under the Precision-Recall Curve (AUPRC)
- Area Under the Reciever Operating Characteristic (AUROC)
- Overall Accurarcy
- Sensitivity + Specificity
https://sinyi-chou.github.io/python-sklearn-precision-recall/
'''

import numpy as np
np.random.seed(401)


def get_AUPRC():
    pass
    

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
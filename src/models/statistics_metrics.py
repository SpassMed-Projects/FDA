from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import numpy as np
np.random.seed(401)


def get_AUPRC(y_test, y_pred):
    average_precision_score(y_test, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall, precision)
    return auc_precision_recall

def get_AUROC(y_test, y_pred):
    auc_roc_score = roc_auc_score(y_test, y_pred)
    return auc_roc_score

def get_Accurarcy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def get_SensitSpecific(y_test, y_pred):
    sensitivity = recall_score(y_test, y_pred,average = 'binary') #TP / (TP + FN)

    #As it was mentioned in the other answers, specificity is the recall of the negative class
    specificity = recall_score(y_test, y_pred, pos_label=0) # TN / (TN + FP) 
    return sensitivity + specificity

def get_Sensitivity(y_test, y_pred):
    sensitivity = recall_score(y_test, y_pred,average = 'binary') #TP / (TP + FN)
    return sensitivity

def get_Specificity(y_test, y_pred):
    specificity = recall_score(y_test, y_pred, average = 'binary', pos_label=0) # TN / (TN + FP) 
    return specificity

def get_Precision(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='binary')
    return precision

def get_F1score(y_test, y_pred):
    score = f1_score(y_test, y_pred, average='binary')
    return score

def get_NPC(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    npc =  tn / (tn + fn)
    return npc

def get_PositiveLR(y_test, y_pred):
    # sensitivity / (1 - specificity)
    positive_lr = get_Sensitivity(y_test, y_pred) / (1 - get_Specificity(y_test, y_pred))
    return positive_lr

def get_NegativeLR(y_test, y_pred):
    # (1 - sensitivity) / specificity
    negative_lr = (1 - get_Sensitivity(y_test, y_pred)) / get_Specificity(y_test, y_pred)
    return negative_lr

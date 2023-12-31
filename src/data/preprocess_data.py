import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm

import csv
from datetime import datetime
from scipy import interpolate

def preprocess_time_string(timeStr):
    if pd.isnull(timeStr):
        return float('nan')
    timeStr = timeStr[:-2]
    date_format = '%Y-%m-%d %H:%M:%S'
    date_obj = datetime.strptime(timeStr, date_format)
    return date_obj

def preprocess_substitute_categories(dictReplacement, dfColumn):
    newCol = []
    for i in range(len(dfColumn)):
        if dfColumn[i] in dictReplacement:
            newCol.append(dictReplacement[dfColumn[i]])
        else: 
            newCol.append(float('nan'))
    return newCol

def imputate_nan_binary(dfColumn):
    p = dfColumn.value_counts()[1] / len(dfColumn)
    for i in range(len(dfColumn)):
        if pd.isnull(dfColumn[i]):
            dfColumn[i] = np.random.binomial(1, p, 1)
    return dfColumn

def generate_dict_from_csv(csvPath,keyName,valueName):
    reader = pd.read_csv(csvPath)
    mydict = {}
    for index, row in reader.iterrows():
        mydict[row[keyName]] = row[valueName]
    return mydict

def impute_with_RF():
    pass

def get_period(startCol,endCol):
    periodCol = []
    length = len(startCol)
    for i in range(length):
        start = startCol[i]
        end = endCol[i]
        if pd.isnull(start) or pd.isnull(end):
            periodCol.append(float('nan'))
        else:
            periodCol.append(end - start)
    return periodCol

def imputate_nan_period_mean(periodCol):
    avg = periodCol.mean()
    for i in range(len(periodCol)):
        if pd.isnull(periodCol[i]):
            periodCol[i] = avg
    return periodCol
        
def imputate_nan_end_date(startCol, endCol, periodCol):
    avg = periodCol.mean()
    for i in range(len(endCol)):
        if pd.isnull(endCol[i]):
            endCol[i] = avg + startCol[i]
    return endCol





import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm

import csv
from datetime import datetime
from scipy import interpolate

def preprocess_time_string(timeStr):
    if pd.isnull(timeStr):
        return None
    timeStr = timeStr[:-2]
    date_format = '%Y-%m-%d %H:%M:%S'
    date_obj = datetime.strptime(timeStr, date_format)
    return date_obj

def preprocess_combine_categories(dictReplacement, dfColumn):
    newCol = []
    for i in range(len(dfColumn)):
        newCol.append(dictReplacement[dfColumn[i]])
    return newCol

def imputate_nan_binary(dfColumn):
    p = dfColumn.value_counts()[1] / len(dfColumn)
    for i in range(len(dfColumn)):
        if pd.isnull(dfColumn[i]):
            dfColumn[i] = np.random.binomial(1, p, 1)
    return dfColumn

def generate_dict_from_csv(csvName,keyName,valueName):
    reader = pd.read_csv(csvName)
    mydict = {}
    for row in reader.iterrows():
        mydict[row[keyName]] = row[valueName]
    return mydict



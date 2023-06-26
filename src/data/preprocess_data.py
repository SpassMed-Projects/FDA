import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm

from datetime import datetime

def preprocess_time_string(timeStr):
    timeStr = timeStr[:-2]
    date_format = '%Y-%m-%d %H:%M:%S'
    date_obj = datetime.strptime(timeStr, date_format)
    return date_obj



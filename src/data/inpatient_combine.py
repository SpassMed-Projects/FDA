import pandas as pd


import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

from datetime import datetime
path = '/home/daisy/FDA_Dataset/demographics_static_clean.csv'
demographic_static = pd.read_csv(path).iloc[:,1:]

path = '/home/daisy/FDA_Dataset/immunization_clean.csv'
immunization = pd.read_csv(path).iloc[:,1:]

path = '/home/daisy/FDA_Dataset/inpatient_medications_ordered_clean.csv'
medications_ordered = pd.read_csv(path).iloc[:,1:]

path = '/home/daisy/FDA_Dataset/inpatient_procedures_clean.csv'
procedures = pd.read_csv(path).iloc[:,1:]
path = '/home/vivi/FDA_datasets/outpatient_visits_preprocessed.csv'
outpatient_visits = pd.read_csv(path).iloc[:,1:]

path ='/home/vivi/FDA_datasets/inpatient_specialty_preprocessed.csv'
inpatient_specialty = pd.read_csv(path).iloc[:,1:]

path ='/home/vivi/FDA_datasets/demographics_event_preprocessed.csv'
demographics_event = pd.read_csv(path).iloc[:,1:]
demographics_event = demographics_event.drop(columns = ['Age at update','State'])

path = '/home/vivi/FDA_datasets/outpatient_state.csv'
outpatient_state = pd.read_csv(path).iloc[:,1:]

path = '/home/vivi/FDA_datasets/inpatient_measurements_bp_preprocessed.csv'
measurements_bp = pd.read_csv(path).iloc[:,1:]
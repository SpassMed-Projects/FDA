import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm

DATA_PATH = "VCHAMPS"

# dataset
# useful column: Internalpatientid, Age at admission, Admission date, 
#                First listed discharge diagnosis icd10 subcategory, Second listed discharge diagnosis icd10 subcategory,
#                Died during admission, Outpatientreferralflag
dataset_path = 'inpatient_admissions_train.csv'
load_path = os.path.join(DATA_PATH, dataset_path)
in_ad = pd.read_csv(load_path)
in_ad = in_ad[["Internalpatientid", "Age at admission", "Admission date",
               "First listed discharge diagnosis icd10 subcategory", "Second listed discharge diagnosis icd10 subcategory",
               "Died during admission", "Outpatientreferralflag"]]

# dataset
# Useful columns: Internalpatientid, Age at ed visit, Ed visit start date, Died during ed visit, First listed diagnosis icd10 subcategory, Second listed diagnosis icd10 subcategory	
dataset_path = 'ed_visits_train.csv'
load_path = os.path.join(DATA_PATH, dataset_path)
ed = pd.read_csv(load_path)
ed = ed[["Internalpatientid", "Age at ed visit", "Died during ed visit", 
         "First listed diagnosis icd10 subcategory", "Second listed diagnosis icd10 subcategory"]]

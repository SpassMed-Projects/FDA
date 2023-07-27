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

path = '/home/hassan/lily/MLA/FDA/inpatient_full_simple.csv'
inpatient = pd.read_csv(path).iloc[:,1:]

path = '/home/hassan/lily/MLA/FDA/inpatient_lab_results.csv'
lab_results = pd.read_csv(path).iloc[:,1:]
lab_results = lab_results.rename(columns = {'count':'lab_count', 'freq':'lab_freq', 
                              'age_mean':'lab_age_mean', 'age_std':'lab_age_std'})

df1 = inpatient.merge(procedures, how = 'left', on = 'Internalpatientid')
df2 = df1.merge(demographic_static, how = 'left', on = 'Internalpatientid')
df3 = df2.merge(immunization, how = 'left', on = 'Internalpatientid')
df4 = df3.merge(medications_ordered, how = 'left', on = 'Internalpatientid')
df5 = df4.merge(inpatient_specialty, how = 'left', on = 'Internalpatientid')
df6 = df5.merge(demographics_event, how = 'left', on = 'Internalpatientid')
df7 = df6.merge(outpatient_state, how = 'left', on = 'Internalpatientid')
df8 = df7.merge(measurements_bp, how = 'left', on = 'Internalpatientid')
df = df8.merge(lab_results, how = 'left', on = 'Internalpatientid')

inpatient_pid = set(inpatient['Internalpatientid'])
medications_pid = set(medications_ordered['Internalpatientid'])
immunization_pid = set(immunization['Internalpatientid'])
procedures_pid = set(procedures['Internalpatientid'])
demographic_static_pid = set(demographic_static['Internalpatientid'])
inpatient_specialty_pid = set(inpatient_specialty['Internalpatientid'])
demographics_event_pid = set(demographics_event['Internalpatientid'])
outpatient_state_pid = set(outpatient_state['Internalpatientid'])
measurements_bp_pid =set(measurements_bp['Internalpatientid'])
lab_results_pid = set(lab_results['Internalpatientid'])

data_pid = [['inpatient_pid',len(list(inpatient_pid)),len(set.intersection(set(inpatient_pid) & set(inpatient_pid)))],
            ['medications_pid',len(list(medications_pid)),len(set.intersection(set(inpatient_pid) & set(medications_pid)))],
            ['immunization_pid',len(list(immunization_pid)),len(set.intersection(set(inpatient_pid) & set(immunization_pid)))],
            ['procedures_pid',len(list(procedures_pid)),len(set.intersection(set(inpatient_pid) & set(procedures_pid)))],
            ['demographic_static_pid',len(list(demographic_static_pid)),len(set.intersection(set(inpatient_pid) & set(demographic_static_pid)))],
            ['inpatient_specialty_pid',len(list(inpatient_specialty_pid)),len(set.intersection(set(inpatient_pid) & set(inpatient_specialty_pid)))],
            ['demographics_event_pid',len(list(demographics_event_pid)),len(set.intersection(set(inpatient_pid) & set(demographics_event_pid)))],
            ['outpatient_state_pid',len(list(outpatient_state_pid)),len(set.intersection(set(inpatient_pid) & set(outpatient_state_pid)))],
            ['measurements_bp_pid',len(list(measurements_bp_pid)),len(set.intersection(set(inpatient_pid) & set(measurements_bp_pid)))],
            ['lab_results_pid',len(list(lab_results_pid)),len(set.intersection(set(inpatient_pid) & set(lab_results_pid)))]]
df_pid = pd.DataFrame(data_pid, columns = ['table name', 'number of unique pid','intersection pid'])
df_pid['percentage'] = df_pid['intersection pid'] / df_pid['number of unique pid'][0] 
df_pid

# Save inpatient data combine other tables (left join)
df1 = df.drop(columns = ['next_readmission_time','Discharge date_y',
                        'Event date','Marital status', 'Marital status encoded',
                        'State','Ruca category','Veteran flag', 'Died', 'AO','Discharge date_x'])

df1.to_csv('/home/daisy/FDA_Dataset/inpatient_all_final_1.csv')

path = '/home/daisy/FDA_Dataset/inpatient_all_final_1.csv'
inpatient_train = pd.read_csv(path).iloc[:,1:]
set(inpatient_train.columns).difference(set(df1.columns))

# Save inpatient combine other test sets for CVD readmission and mortality
df_CVD = df.drop(columns = ['Event date','Marital status', 'Marital status encoded',
                        'State','Ruca category','Veteran flag', 'Died', 'AO',
                        'Discharge date_y', 'Discharge date_x'])
df_CVD = df_CVD[df_CVD['next_readmission_time'] > 0]
df_CVD = df_CVD.drop(columns = ['next_readmission_time'])

# CVD_readmission: readmitted beacuse of CVD
# next_readmission_time: time difference between this admission and next admission
df_CVD.to_csv('/home/daisy/FDA_Dataset/inpatient_CVD_final_1.csv')

df1 = inpatient.merge(procedures, how = 'inner', on = 'Internalpatientid')
df2 = df1.merge(demographic_static, how = 'inner', on = 'Internalpatientid')
df3 = df2.merge(immunization, how = 'inner', on = 'Internalpatientid')
df4 = df3.merge(medications_ordered, how = 'inner', on = 'Internalpatientid')

df5 = df4.merge(inpatient_specialty, how = 'inner', on = 'Internalpatientid')
df6 = df5.merge(demographics_event, how = 'inner', on = 'Internalpatientid')
df7 = df6.merge(outpatient_state, how = 'inner', on = 'Internalpatientid')
df8 = df7.merge(measurements_bp, how = 'inner', on = 'Internalpatientid')

df_2 = df8.merge(lab_results, how = 'inner', on = 'Internalpatientid')
# Save inpatient data for all cause with inner join
df_2_all = df_2.drop(columns = ['next_readmission_time','Event date','Marital status', 
                                'Marital status encoded','State','Ruca category',
                                'Veteran flag', 'Died', 'AO','Discharge date_y',
                                'Discharge date_x'])


df_2_all.to_csv('/home/daisy/FDA_Dataset/inpatient_all_final_2.csv')



df_CVD_2 = df_2.drop(columns = ['Event date','Marital status', 'Marital status encoded',
                        'State','Ruca category','Veteran flag', 'Died', 'AO',
                        'Discharge date_y', 'Discharge date_x'])
df_CVD_2 = df_CVD_2[df_CVD_2['next_readmission_time'] > 0] 
df_CVD_2 = df_CVD_2.drop(columns = ['next_readmission_time'])

# CVD_readmission: readmitted beacuse of CVD
# next_readmission_time: time difference between this admission and next admission
df_CVD_2.to_csv('/home/daisy/FDA_Dataset/inpatient_CVD_final_2.csv')

path = '/home/daisy/FDA_Dataset/inpatient_CVD_final_test_1.csv'
test = pd.read_csv(path).iloc[:,1:]
train = pd.read_csv('/home/daisy/FDA_Dataset/inpatient_CVD_final_1.csv').iloc[:,1:]



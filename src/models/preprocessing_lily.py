import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math

# use as a parameter in class
cardiovascular = ["Heart failure, unspecified", "Other heart failure",
                  "Cardiogenic shock", "Hypertensive heart disease with heart failure",
                  "Hypertensive heart and kidney disease with heart failure", "Unstable angina", "Other forms of chronic ischemic heart disease",
                  "Atherosclerotic heart disease of native coronary artery", "Atrial fibrillation", "Atrial flutter",
                  "Supraventricular tachycardia", "Ventricular tachycardia"]

sub_abuse_and_mental = ['DRUG DEPENDENCE TRMT UNIT', 'PSYCHIATRIC MENTALLY INFIRM', 'SUBSTANCE ABUSE RES TRMT PROG', 'PLASTIC SURGERY', 'PSYCH RESID REHAB TRMT PROG', 'SUBSTANCE ABUSE INTERMED CARE', 'ACUTE PSYCHIATRY (<45 DAYS)', 'DOMICILIARY PTSD', 
                        'ALCOHOL DEPENDENCE TRMT UNIT', 'EVAL/BRF TRMT PTSD UNIT(EBTPU)', 'PTSD RESID REHAB PROG', 'PSYCH RESID REHAB PROG', 'PTSD CWT/TR', 
                        'PTSD RESIDENTIAL REHAB PROG', 'SUBST ABUSE STAR I, II & III', 'SUBSTANCE ABUSE TRMT UNIT', 'GEN INTERMEDIATE PSYCH', 'LONG TERM PSYCHIATRY(>45 DAYS)', 'SUBSTANCE ABUSE RESID PROG', 
                        'DOMICILIARY SUBSTANCE ABUSE', 'SIPU (SPEC INPT PTSD UNIT)', 'ZZALCOHOL DEPENDENCE TRMT UNIT', 'ZZSUBSTANCE ABUSE INTERMEDCARE', 'PSYCHIATRY', 'ZZDRUG DEPENDENCE TRMT UNIT',
                        'ZZSUBSTANCE ABUSE TRMT UNIT', 'GEN MEDICINE (ACUTE)', 'zSUBST ABUSE STAR I, II & III', 'ZZSUBST ABUSE STAR I,II,II', 'SUBST ABUSE CWT/TRANS RESID', 'HIGH INTENSITY GEN PSYCH INPAT', 'HALFWAY HOUSE']
medical = ['HEMATOLOGY/ONCOLOGY', 'GASTROENTEROLOGY', 'INTERMEDIATE MEDICINE', 'ANESTHESIOLOGY', 'PROCTOLOGY', 'CARDIAC SURGERY', 'TRANSPLANTATION', 'CARDIOLOGY', 'METABOLIC',
           'GENERAL(ACUTE MEDICINE)', 'PEDIATRICS', 'VASCULAR', 'OPHTHALMOLOGY', 'NEUROSURGERY', 'SURGICAL STEPDOWN', 'UROLOGY', 'PULMONARY, TUBERCULOSIS', 'PERIPHERAL VASCULAR', 
           'THORACIC SURGERY', 'MEDICAL STEP DOWN', 'GENERAL SURGERY', 'PULMONARY, NON-TB', 'EPILEPSY CENTER', 'NEUROLOGY', 'SPINAL CORD INJURY', 'ORAL SURGERY',
           'PODIATRY', 'EAR, NOSE, THROAT (ENT)', 'ENDOCRINOLOGY', 'CARDIAC-STEP DOWN UNIT', 'TELEMETRY', 'OB/GYN', 'ORTHOPEDIC', 'DOMICILIARY','ALLERGY', 'STROKE UNIT', 'DERMATOLOGY',
           'CARDIAC INTENSIVE CARE UNIT', 'HOSPICE FOR ACUTE CARE', 'SURGICAL ICU', 'MEDICAL ICU']
rehab = ['NH SHORT STAY REHABILITATION', 'BLIND REHAB OBSERVATION', 'BLIND REHAB', 'NH LONG STAY DEMENTIA CARE', 'NH SHORT-STAY CONTINUING CARE', 'NH HOSPICE', 'NEUROLOGY OBSERVATION', 'NH LONG-STAY MH RECOVERY', 
         'NH SHORT-STAY MH RECOVERY', 'NH SHORT STAY RESTORATIVE', 'REHABILITATION MEDICINE','NH SHORT STAY DEMENTIA CARE', 'RESPITE CARE (MEDICINE)', 'PM&R TRANSITIONAL REHAB', 
         'SPINAL CORD INJURY OBSERVATION', 'POLYTRAUMA REHAB UNIT', 'SURGICAL OBSERVATION', 'NHCU', 'NH SHORT STAY SKILLED NURSING', 'NH GEM NURSING HOME CARE', 'NH LONG STAY SKILLED NURSING', 'NH LONG-STAY CONTINUING CARE', 'ED OBSERVATION', 
         'MEDICAL OBSERVATION', 'REHAB MEDICINE OBSERVATION', 'PSYCHIATRIC OBSERVATION', 'NH LONG STAY SPINAL CORD INJ', 'DOD BEDS IN VA FACILITY', 'NON-DOD BEDS IN VA FACILITY', 'GENERAL CWT/TR', 'HOMELESS CWT/TRANS RESID', 'PRRTP', 'HIGH INTENSITY GEN INPT',
         'DOMICILIARY CHV', 'STAR I, II & III']
gem = ['GRECC-MED', 'SHORT STAY GRECC-GEM-NHCU', 'GEM DOMICILIARY', 'SHORT STAY GRECC-NHCU', 'GEM REHABILITATION MEDICINE', 'GRECC-GEM-REHAB', 'GEM NEUROLOGY', 
       'GEM ACUTE MEDICINE', 'LONG STAY GRECC-NHCU', 'GEM PSYCHIATRIC BEDS', 'GERONTOLOGY', 'GEM INTERMEDIATE CARE']
others = ['Not specified', '(Censored)', 'Not specified (no value)']

def preprocess_inpatient_readmission(datatype):
    '''
    datatype: ["train", "test"]
    return: datapath
    train: /home/hassan/lily/MLA/FDA/inpatient_full_simple.csv
    test: /home/hassan/lily/MLA/FDA/inpatient_simple_test.csv

    for preprocessing the data for targets: All-cause readmission, CVD Readmission
    '''
    if datatype == "train": 
        DATA_PATH = "/home/bhatti/dataset/VCHAMPS"
        dataset_path = 'inpatient_admissions_train.csv'
        target_path = '/home/hassan/lily/MLA/FDA/inpatient_full_simple.csv'

        datapath = "/home/bhatti/dataset/VCHAMPS/death_train.csv"
        death = pd.read_csv(datapath, index_col=0)
    else: 
        DATA_PATH = '/data/public/MLA/VCHAMPS-Test/'
        dataset_path = "inpatient_admissions_test.csv"
        target_path = '/home/hassan/lily/MLA/FDA/inpatient_simple_test.csv'

        datapath = "/data/public/MLA/VCHAMPS-Test/death_test.csv"
        death = pd.read_csv(datapath, index_col=0)

    load_path = os.path.join(DATA_PATH, dataset_path)
    in_ad = pd.read_csv(load_path,index_col=0)

    # Died at location
    in_ad["Died during admission"] = in_ad["Died during admission"].replace({"Yes":1, "No": 0})

    # num transfers
    in_ad['Transfer'] = [0] * len(in_ad)
    def transfer(data):
        data.loc[data['Admitting unit service'] != data['Discharging unit service'], 'Transfer'] = 1
        return data
    in_ad = transfer(in_ad)

    # regrouping admitting unit
    in_ad["Admitting unit service"] = in_ad["Admitting unit service"].replace({'REHAB MEDICINE':'OTHERS', 'BLIND REHAB':'OTHERS',
                                        '(Censored)':'NON-COUNT', 'Not specified (no value)':'NON-COUNT', 'Not specified':'NON-COUNT',
                                        'INTERMEDIATE MED':'OTHERS', 'NEUROLOGY':'OTHERS', 'SPINAL CORD INJURY':'OTHERS'})
    in_ad = pd.concat([in_ad, pd.get_dummies(in_ad['Admitting unit service'])], axis=1)

    # length of stay
    in_ad["Admission date"] = pd.to_datetime(in_ad["Admission date"], format='%Y-%m-%d %H:%M:%S.%f')
    in_ad["Discharge date"] = pd.to_datetime(in_ad["Discharge date"], format='%Y-%m-%d %H:%M:%S.%f')
    in_ad["stay_length"] = (in_ad["Discharge date"] - in_ad["Admission date"]).dt.days + round((in_ad["Discharge date"] - in_ad["Admission date"]).dt.seconds/3600/24,2)

    # imputation
    mean_stay_length = in_ad.groupby("Admitting unit service")["stay_length"].mean()
    nan_admission = in_ad[in_ad["stay_length"].isna()]
    in_ad = in_ad[~in_ad["stay_length"].isna()]

    admission_stay_length = []
    for i, row in nan_admission.iterrows():
        service = row["Admitting unit service"]
        diff = mean_stay_length[service]
        
        admission_stay_length.append(round(diff,2))
    nan_admission["stay_length"] = admission_stay_length

    in_ad = pd.concat([in_ad, nan_admission])

    # age at admission
    in_ad['Age 20-40'] = [0] * len(in_ad)
    in_ad['Age 40-60'] = [0] * len(in_ad)
    in_ad['Age 60-80'] = [0] * len(in_ad)
    in_ad['Age 80-100'] = [0] * len(in_ad)
    in_ad['Age 100-120'] = [0] * len(in_ad)
    def age_category(data):
        data.loc[(data['Age at admission'] > 20) & (data['Age at admission'] <= 40), 'Age 20-40'] = 1
        data.loc[(data['Age at admission'] > 40) & (data['Age at admission'] <= 60), 'Age 40-60'] = 1
        data.loc[(data['Age at admission'] > 60) & (data['Age at admission'] <= 80), 'Age 60-80'] = 1
        data.loc[(data['Age at admission'] > 80) & (data['Age at admission'] <= 100), 'Age 80-100'] = 1
        data.loc[(data['Age at admission'] > 100) & (data['Age at admission'] <= 120), 'Age 100-120'] = 1
        return data
    in_ad = age_category(in_ad)

    # cvd
    in_ad["cd_diagnosis"] = [0] * len(in_ad)
    def cd_diagnosis(data):
        data.loc[(in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains('|'.join(cardiovascular))
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("Systolic (congestive) heart failure", regex=False)
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("Diastolic (congestive) heart failure", regex=False)
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("Combined systolic (congestive) and diastolic (congestive) heart failure", regex=False)
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("ST elevation (STEMI) myocardial infarction", regex=False)
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("Non-ST elevation (NSTEMI) myocardial infarction", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains('|'.join(cardiovascular))
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("Systolic (congestive) heart failure", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("Diastolic (congestive) heart failure", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("Combined systolic (congestive) and diastolic (congestive) heart failure", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("ST elevation (STEMI) myocardial infarction", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("Non-ST elevation (NSTEMI) myocardial infarction", regex=False)), "cd_diagnosis"] = 1
        return data
    in_ad = cd_diagnosis(in_ad)

    # specialty
    in_ad['Mental'] = [0] * len(in_ad)
    in_ad['Medical'] = [0] * len(in_ad)
    in_ad['Rehab'] = [0] * len(in_ad)
    in_ad['Gerontology'] = [0] * len(in_ad)
    in_ad['Others_Specialty'] = [0] * len(in_ad)
    def age_category(data):
        data.loc[(data['Admitting specialty'].isin(sub_abuse_and_mental)), 'Mental'] = 1
        data.loc[(data['Admitting specialty'].isin(medical)), 'Medical'] = 1
        data.loc[(data['Admitting specialty'].isin(rehab)), 'Rehab'] = 1
        data.loc[(data['Admitting specialty'].isin(gem)), 'Gerontology'] = 1
        data.loc[(data['Admitting specialty'].isin(others)), 'Others_Specialty'] = 1
        return data
    in_ad = age_category(in_ad)

    # final dataset
    def final_set_before(ids, group):
        
        readmission = 0
        num_admissions = group["Age at admission"].nunique()
        if num_admissions > 1: readmission=1

        age_mean = group["Age at admission"].mean()
        age_std = group["Age at admission"].std()
        if group["Age at admission"].nunique() == 1: age_std = 0

        min_age = group["Age at admission"].min()
        max_age = group["Age at admission"].max()

        freq = num_admissions/(math.floor(max_age - min_age) + 1)

        min_stay = group["stay_length"].min()
        max_stay = group["stay_length"].min()
        stay_mean = group["stay_length"].mean()
        stay_std = group["stay_length"].std()
        if group["stay_length"].nunique() == 1: stay_std = 0

        # num_cvd_readmission = max(0, group['cd_diagnosis'].sum() - 1)
        num_cvd_admission = group['cd_diagnosis'].sum()

        cvd = 0
        if group["cd_diagnosis"].sum() > 0: cvd = 1
        
        Died = 0
        if group["Died during admission"].sum() > 0: Died = 1

        AO = 0
        if group["Agentorangeflag"].sum() > 0: AO = 1

        df = pd.DataFrame(data={'Internalpatientid': [ids], 'num_stays': [num_admissions], 'stay_length': group["stay_length"].sum(),
                                'num_unique_units': group["Admitting unit service"].nunique(), "num_transfers": group["Transfer"].sum(), 
                                "num_cvd_admission": [num_cvd_admission], "Died": [Died], "AO": [AO], "CVD": [cvd],
                                "unique_admitting_specialty": group["Admitting specialty"].nunique(), "unique_discharging_specialty": group["Discharging specialty"].nunique(),
                                "DOMICILIARY": group["DOMICILIARY"].sum(), "MEDICINE": group["MEDICINE"].sum(), "NHCU":group["NHCU"].sum(),
                                "NON-COUNT":group["NON-COUNT"].sum(), "OTHERS":group["OTHERS"].sum(), 'PSYCHIATRY': group['PSYCHIATRY'].sum(), 'SURGERY': group['SURGERY'].sum(),
                                'Age 20-40': group["Age 20-40"].sum(), 'Age 40-60': group["Age 40-60"].sum(), 'Age 60-80':group["Age 60-80"].sum(), 
                                'Age 80-100':group["Age 80-100"].sum(), 'Age 100-120':group["Age 100-120"].sum(), "age_mean": [age_mean], "age_std": [age_std], 
                                "age_min": [min_age], "age_max": [max_age], "stay_min": [min_stay], "stay_max": [max_stay], "stay_mean": [stay_mean],
                                "stay_std": [stay_std], "freq": [round(freq,2)], 'Medical': group["Medical"].sum(), 'Mental':group["Mental"].sum(), 
                                'Others_Specialty':group["Others_Specialty"].sum(), 'Rehab': group["Rehab"].sum(), 'Gerontology': group["Gerontology"].sum()
                                })
        df = df.reset_index(drop=True)
        return df
    
    def final_set(ids, group):
        full = []
        ddl = 0
        for i in range(len(group)):
            if ((i+1) != (len(group) - 1)) and (len(group) != 1): 
                # for now we only consider the case when the last row is used as the indicator
                continue
            df = final_set_before(ids, group.iloc[0:i+1,])
            
            CVD_readmission = 0

            if len(group) == i+1: 
                read_time = 0
                ddl = group.iloc[i]["Discharge date"]
            else: 
                start = group.iloc[i]["Discharge date"]
                end = group.iloc[i+1]["Admission date"]
                ddl = start

                start = pd.to_datetime(start,
                format='%Y-%m-%d %H:%M:%S.%f')
                end = pd.to_datetime(end,
                format='%Y-%m-%d %H:%M:%S.%f')
                read_time = pd.Timedelta(end - start).days
                read_time += round(pd.Timedelta(end - start).seconds/3600/24,2)

                if df["num_cvd_admission"].item() > 0:
                    # if this patient once admitted due to CVD
                    if group.iloc[i+1]["cd_diagnosis"] == 1:
                        # if the indicator admission also caused by CVD
                        CVD_readmission = 1

            df["CVD_readmission"] = CVD_readmission
            df["next_readmission_time"] = read_time
            df["Discharge date"] = ddl
            # if read_time < 0: print(group[["Admission date","Discharge date"]])

            if (read_time > 300) or (read_time == 0): threshold = 0
            else: threshold = 1
            df["readmission within 300 days"] = threshold
            full.append(df)
        return pd.concat(full)
    
    tidy_dataset = []
    for ids, group in tqdm(in_ad.groupby("Internalpatientid")):
        group = group.sort_values(by = ['Admission date'],ascending=True).reset_index(drop = True)
        
        df = final_set(ids, group)
        tidy_dataset.append(df)
        
    tidy_dataset = pd.concat(tidy_dataset)

    if datatype == "train": 
        tidy_dataset.to_csv("/home/hassan/lily/MLA/FDA/inpatient_full_simple.csv")
        return "/home/hassan/lily/MLA/FDA/inpatient_full_simple.csv"
    
    if datatype == "test": 
        tidy_dataset.to_csv("/home/hassan/lily/MLA/FDA/inpatient_simple_test.csv")
        return "/home/hassan/lily/MLA/FDA/inpatient_simple_test.csv"




def preprocess_inpatient_cvd_mortality(datatype):
    '''
    datatype: ["train", "test"]
    return: datapath
    train: /home/hassan/lily/MLA/FDA/inpatient_cvd_mortality.csv
    test: /home/hassan/lily/MLA/FDA/inpatient_cvd_mortality_test.csv

    for preprocessing the data for targets: CVD mortality
    '''
    if datatype == "train": 
        DATA_PATH = "/home/bhatti/dataset/VCHAMPS"
        dataset_path = 'inpatient_admissions_train.csv'
        target_path = '/home/hassan/lily/MLA/FDA/inpatient_full_simple.csv'

        datapath = "/home/bhatti/dataset/VCHAMPS/death_train.csv"
        death = pd.read_csv(datapath, index_col=0)
    else: 
        DATA_PATH = '/data/public/MLA/VCHAMPS-Test/'
        dataset_path = "inpatient_admissions_test.csv"
        target_path = '/home/hassan/lily/MLA/FDA/inpatient_simple_test.csv'

        datapath = "/data/public/MLA/VCHAMPS-Test/death_test.csv"
        death = pd.read_csv(datapath, index_col=0)

    load_path = os.path.join(DATA_PATH, dataset_path)
    in_ad = pd.read_csv(load_path,index_col=0)

    # Died at location
    in_ad["Died during admission"] = in_ad["Died during admission"].replace({"Yes":1, "No": 0})

    # num transfers
    in_ad['Transfer'] = [0] * len(in_ad)
    def transfer(data):
        data.loc[data['Admitting unit service'] != data['Discharging unit service'], 'Transfer'] = 1
        return data
    in_ad = transfer(in_ad)

    # regrouping admitting unit
    in_ad["Admitting unit service"] = in_ad["Admitting unit service"].replace({'REHAB MEDICINE':'OTHERS', 'BLIND REHAB':'OTHERS',
                                        '(Censored)':'NON-COUNT', 'Not specified (no value)':'NON-COUNT', 'Not specified':'NON-COUNT',
                                        'INTERMEDIATE MED':'OTHERS', 'NEUROLOGY':'OTHERS', 'SPINAL CORD INJURY':'OTHERS'})
    in_ad = pd.concat([in_ad, pd.get_dummies(in_ad['Admitting unit service'])], axis=1)

    # length of stay
    in_ad["Admission date"] = pd.to_datetime(in_ad["Admission date"], format='%Y-%m-%d %H:%M:%S.%f')
    in_ad["Discharge date"] = pd.to_datetime(in_ad["Discharge date"], format='%Y-%m-%d %H:%M:%S.%f')
    in_ad["stay_length"] = (in_ad["Discharge date"] - in_ad["Admission date"]).dt.days + round((in_ad["Discharge date"] - in_ad["Admission date"]).dt.seconds/3600/24,2)

    # imputation
    mean_stay_length = in_ad.groupby("Admitting unit service")["stay_length"].mean()
    nan_admission = in_ad[in_ad["stay_length"].isna()]
    in_ad = in_ad[~in_ad["stay_length"].isna()]

    admission_stay_length = []
    for i, row in nan_admission.iterrows():
        service = row["Admitting unit service"]
        diff = mean_stay_length[service]
        
        admission_stay_length.append(round(diff,2))
    nan_admission["stay_length"] = admission_stay_length

    in_ad = pd.concat([in_ad, nan_admission])

    # age at admission
    in_ad['Age 20-40'] = [0] * len(in_ad)
    in_ad['Age 40-60'] = [0] * len(in_ad)
    in_ad['Age 60-80'] = [0] * len(in_ad)
    in_ad['Age 80-100'] = [0] * len(in_ad)
    in_ad['Age 100-120'] = [0] * len(in_ad)
    def age_category(data):
        data.loc[(data['Age at admission'] > 20) & (data['Age at admission'] <= 40), 'Age 20-40'] = 1
        data.loc[(data['Age at admission'] > 40) & (data['Age at admission'] <= 60), 'Age 40-60'] = 1
        data.loc[(data['Age at admission'] > 60) & (data['Age at admission'] <= 80), 'Age 60-80'] = 1
        data.loc[(data['Age at admission'] > 80) & (data['Age at admission'] <= 100), 'Age 80-100'] = 1
        data.loc[(data['Age at admission'] > 100) & (data['Age at admission'] <= 120), 'Age 100-120'] = 1
        return data
    in_ad = age_category(in_ad)

    # cvd
    in_ad["cd_diagnosis"] = [0] * len(in_ad)
    def cd_diagnosis(data):
        data.loc[(in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains('|'.join(cardiovascular))
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("Systolic (congestive) heart failure", regex=False)
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("Diastolic (congestive) heart failure", regex=False)
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("Combined systolic (congestive) and diastolic (congestive) heart failure", regex=False)
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("ST elevation (STEMI) myocardial infarction", regex=False)
        | in_ad["Second listed discharge diagnosis icd10 subcategory"].str.contains("Non-ST elevation (NSTEMI) myocardial infarction", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains('|'.join(cardiovascular))
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("Systolic (congestive) heart failure", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("Diastolic (congestive) heart failure", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("Combined systolic (congestive) and diastolic (congestive) heart failure", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("ST elevation (STEMI) myocardial infarction", regex=False)
        | in_ad["First listed discharge diagnosis icd10 subcategory"].str.contains("Non-ST elevation (NSTEMI) myocardial infarction", regex=False)), "cd_diagnosis"] = 1
        return data
    in_ad = cd_diagnosis(in_ad)

    # specialty
    in_ad['Mental'] = [0] * len(in_ad)
    in_ad['Medical'] = [0] * len(in_ad)
    in_ad['Rehab'] = [0] * len(in_ad)
    in_ad['Gerontology'] = [0] * len(in_ad)
    in_ad['Others_Specialty'] = [0] * len(in_ad)
    def age_category(data):
        data.loc[(data['Admitting specialty'].isin(sub_abuse_and_mental)), 'Mental'] = 1
        data.loc[(data['Admitting specialty'].isin(medical)), 'Medical'] = 1
        data.loc[(data['Admitting specialty'].isin(rehab)), 'Rehab'] = 1
        data.loc[(data['Admitting specialty'].isin(gem)), 'Gerontology'] = 1
        data.loc[(data['Admitting specialty'].isin(others)), 'Others_Specialty'] = 1
        return data
    in_ad = age_category(in_ad)

    # final dataset
    def final_set_before(ids, group):
        
        readmission = 0
        num_admissions = group["Age at admission"].nunique()
        if num_admissions > 1: readmission=1

        age_mean = group["Age at admission"].mean()
        age_std = group["Age at admission"].std()
        if group["Age at admission"].nunique() == 1: age_std = 0

        min_age = group["Age at admission"].min()
        max_age = group["Age at admission"].max()

        freq = num_admissions/(math.floor(max_age - min_age) + 1)

        min_stay = group["stay_length"].min()
        max_stay = group["stay_length"].min()
        stay_mean = group["stay_length"].mean()
        stay_std = group["stay_length"].std()
        if group["stay_length"].nunique() == 1: stay_std = 0

        # num_cvd_readmission = max(0, group['cd_diagnosis'].sum() - 1)
        num_cvd_admission = group['cd_diagnosis'].sum()

        cvd = 0
        if group["cd_diagnosis"].sum() > 0: cvd = 1
        
        Died = 0
        if group["Died during admission"].sum() > 0: Died = 1

        AO = 0
        if group["Agentorangeflag"].sum() > 0: AO = 1

        if (group.iloc[len(group)-1]["Died during admission"] == 1) and (group.iloc[len(group)-1]['cd_diagnosis'] == 1): died_by_cvd = 1
        else: died_by_cvd = 0

        df = pd.DataFrame(data={'Internalpatientid': [ids], 'num_stays': [num_admissions], 'stay_length': group["stay_length"].sum(),
                                'num_unique_units': group["Admitting unit service"].nunique(), "num_transfers": group["Transfer"].sum(), 
                                "num_cvd_admission": [num_cvd_admission], "Died": [Died], "AO": [AO], "CVD": [cvd],
                                "unique_admitting_specialty": group["Admitting specialty"].nunique(), "unique_discharging_specialty": group["Discharging specialty"].nunique(),
                                "DOMICILIARY": group["DOMICILIARY"].sum(), "MEDICINE": group["MEDICINE"].sum(), "NHCU":group["NHCU"].sum(),
                                "NON-COUNT":group["NON-COUNT"].sum(), "OTHERS":group["OTHERS"].sum(), 'PSYCHIATRY': group['PSYCHIATRY'].sum(), 'SURGERY': group['SURGERY'].sum(),
                                'Age 20-40': group["Age 20-40"].sum(), 'Age 40-60': group["Age 40-60"].sum(), 'Age 60-80':group["Age 60-80"].sum(), 
                                'Age 80-100':group["Age 80-100"].sum(), 'Age 100-120':group["Age 100-120"].sum(), "age_mean": [age_mean], "age_std": [age_std], 
                                "age_min": [min_age], "age_max": [max_age], "stay_min": [min_stay], "stay_max": [max_stay], "stay_mean": [stay_mean],
                                "stay_std": [stay_std], "freq": [round(freq,2)], 'Medical': group["Medical"].sum(), 'Mental':group["Mental"].sum(), 
                                'Others_Specialty':group["Others_Specialty"].sum(), 'Rehab': group["Rehab"].sum(), 'Gerontology': group["Gerontology"].sum(),
                                "died_by_cvd": died_by_cvd
                                })
        df = df.reset_index(drop=True)
        return df
    
    tidy_dataset = []
    for ids, group in tqdm(in_ad.groupby("Internalpatientid")):
        group = group.sort_values(by = ['Admission date'],ascending=True).reset_index(drop = True)
        df = final_set_before(ids, group)
        tidy_dataset.append(df)
        
    tidy_dataset = pd.concat(tidy_dataset)

    tidy_dataset = tidy_dataset.merge(death, how="left",on="Internalpatientid")

    # we only want dead patients
    tidy_dataset["Died"] = [1] * len(tidy_dataset)
    tidy_dataset.loc[tidy_dataset["Age at death"].isna(), "Died"] = 0
    tidy_dataset = tidy_dataset[tidy_dataset["Died"] == 1]

    tidy_dataset = tidy_dataset.drop(["Died", "Death date"], axis=1)

    if datatype == "train": tidy_dataset.to_csv("/home/hassan/lily/MLA/FDA/inpatient_cvd_mortality.csv")
    if datatype == "test": tidy_dataset.to_csv("/home/hassan/lily/MLA/FDA/inpatient_cvd_mortality_test.csv")

    # add ed_visits information
    if datatype == "train": 
        in_ad = pd.read_csv("/home/bhatti/dataset/VCHAMPS/inpatient_admissions_train.csv", index_col=0)
        ed = pd.read_csv("/home/bhatti/dataset/VCHAMPS/ed_visits_train.csv", index_col=0)
        death = pd.read_csv("/home/bhatti/dataset/VCHAMPS/death_train.csv", index_col=0)
    if datatype == "test": 
        in_ad = pd.read_csv("/data/public/MLA/VCHAMPS-Test/inpatient_admissions_test.csv", index_col=0)
        ed = pd.read_csv("/data/public/MLA/VCHAMPS-Test/ed_visits_test.csv", index_col=0)
        death = pd.read_csv("/data/public/MLA/VCHAMPS-Test/death_test.csv", index_col=0)
    
    ed = ed.rename(columns={'Age at ed visit': 'Age at admission', 'Ed visit start date': 'Admission date', 
    'Discharge date ed': 'Discharge date', 'Died during ed visit': 'Died during admission',
    'First listed diagnosis icd10 subcategory': 'First listed discharge diagnosis icd10 subcategory',
    'Second listed diagnosis icd10 subcategory': 'Second listed discharge diagnosis icd10 subcategory'})

    full = pd.concat([ed, in_ad[['Internalpatientid', 'Age at admission', 'Admission date',
        'Discharge date', 'Died during admission',
        'First listed discharge diagnosis icd10 subcategory',
        'Second listed discharge diagnosis icd10 subcategory', 'State']]])
    
    # died
    full["Died during admission"] = full["Died during admission"].replace({"Yes":1, "No": 0})

    # stay length
    full["Admission date"] = pd.to_datetime(full["Admission date"], format='%Y-%m-%d %H:%M:%S.%f')
    full["Discharge date"] = pd.to_datetime(full["Discharge date"], format='%Y-%m-%d %H:%M:%S.%f')
    full["stay_length"] = (full["Discharge date"] - full["Admission date"]).dt.days + round((full["Discharge date"] - full["Admission date"]).dt.seconds/3600/24,2)

    full["stay_length"] = full["stay_length"].fillna(full["stay_length"].mean())

    # age
    full['Age 20-40'] = [0] * len(full)
    full['Age 40-60'] = [0] * len(full)
    full['Age 60-80'] = [0] * len(full)
    full['Age 80-100'] = [0] * len(full)
    full['Age 100-120'] = [0] * len(full)
    def age_category(data):
        data.loc[(data['Age at admission'] > 20) & (data['Age at admission'] <= 40), 'Age 20-40'] = 1
        data.loc[(data['Age at admission'] > 40) & (data['Age at admission'] <= 60), 'Age 40-60'] = 1
        data.loc[(data['Age at admission'] > 60) & (data['Age at admission'] <= 80), 'Age 60-80'] = 1
        data.loc[(data['Age at admission'] > 80) & (data['Age at admission'] <= 100), 'Age 80-100'] = 1
        data.loc[(data['Age at admission'] > 100) & (data['Age at admission'] <= 120), 'Age 100-120'] = 1
        return data
    full = age_category(full)
    
    # cvd
    full["cd_diagnosis"] = [0] * len(full)
    def cd_diagnosis(data):
        data.loc[(full["Second listed discharge diagnosis icd10 subcategory"].str.contains('|'.join(cardiovascular))
        | full["Second listed discharge diagnosis icd10 subcategory"].str.contains("Systolic (congestive) heart failure", regex=False)
        | full["Second listed discharge diagnosis icd10 subcategory"].str.contains("Diastolic (congestive) heart failure", regex=False)
        | full["Second listed discharge diagnosis icd10 subcategory"].str.contains("Combined systolic (congestive) and diastolic (congestive) heart failure", regex=False)
        | full["Second listed discharge diagnosis icd10 subcategory"].str.contains("ST elevation (STEMI) myocardial infarction", regex=False)
        | full["Second listed discharge diagnosis icd10 subcategory"].str.contains("Non-ST elevation (NSTEMI) myocardial infarction", regex=False)
        | full["First listed discharge diagnosis icd10 subcategory"].str.contains('|'.join(cardiovascular))
        | full["First listed discharge diagnosis icd10 subcategory"].str.contains("Systolic (congestive) heart failure", regex=False)
        | full["First listed discharge diagnosis icd10 subcategory"].str.contains("Diastolic (congestive) heart failure", regex=False)
        | full["First listed discharge diagnosis icd10 subcategory"].str.contains("Combined systolic (congestive) and diastolic (congestive) heart failure", regex=False)
        | full["First listed discharge diagnosis icd10 subcategory"].str.contains("ST elevation (STEMI) myocardial infarction", regex=False)
        | full["First listed discharge diagnosis icd10 subcategory"].str.contains("Non-ST elevation (NSTEMI) myocardial infarction", regex=False)), "cd_diagnosis"] = 1
        return data
    full = cd_diagnosis(full)

    # final dataset
    def final_set_before(ids, group):
        
        readmission = 0
        num_admissions = group["Age at admission"].nunique()
        if num_admissions > 1: readmission=1

        age_mean = group["Age at admission"].mean()
        age_std = group["Age at admission"].std()
        if group["Age at admission"].nunique() == 1: age_std = 0

        min_age = group["Age at admission"].min()
        max_age = group["Age at admission"].max()

        freq = num_admissions/(math.floor(max_age - min_age) + 1)

        min_stay = group["stay_length"].min()
        max_stay = group["stay_length"].min()
        stay_mean = group["stay_length"].mean()
        stay_std = group["stay_length"].std()
        if group["stay_length"].nunique() == 1: stay_std = 0

        # num_cvd_readmission = max(0, group['cd_diagnosis'].sum() - 1)
        num_cvd_admission = group['cd_diagnosis'].sum()

        cvd = 0
        if group["cd_diagnosis"].sum() > 0: cvd = 1

        if (group.iloc[len(group)-1]["Died during admission"] == 1) and (group.iloc[len(group)-1]['cd_diagnosis'] == 1): died_by_cvd = 1
        else: died_by_cvd = 0

        df = pd.DataFrame(data={'Internalpatientid': [ids], 'num_stays': [num_admissions], 'stay_length': group["stay_length"].sum(), 
                                "num_cvd_admission": [num_cvd_admission], "CVD": [cvd],
                                'Age 20-40': group["Age 20-40"].sum(), 'Age 40-60': group["Age 40-60"].sum(), 'Age 60-80':group["Age 60-80"].sum(), 
                                'Age 80-100':group["Age 80-100"].sum(), 'Age 100-120':group["Age 100-120"].sum(), "age_mean": [age_mean], "age_std": [age_std], 
                                "age_min": [min_age], "age_max": [max_age], "stay_min": [min_stay], "stay_max": [max_stay], "stay_mean": [stay_mean],
                                "stay_std": [stay_std], "freq": [round(freq,2)], "died_by_cvd": died_by_cvd
                                })
        df = df.reset_index(drop=True)
        return df
    
    tidy_dataset = []
    for ids, group in tqdm(full.groupby("Internalpatientid")):
        group = group.sort_values(by = ['Admission date'],ascending=True).reset_index(drop = True)
        df = final_set_before(ids, group)
        tidy_dataset.append(df)
        
    tidy_dataset = pd.concat(tidy_dataset)

    if datatype == "train": inpatient_before = pd.read_csv("/home/hassan/lily/MLA/FDA/inpatient_cvd_mortality.csv", index_col = 0)
    if datatype == "test": inpatient_before = pd.read_csv("/home/hassan/lily/MLA/FDA/inpatient_cvd_mortality_test.csv", index_col = 0)

    inpatient_before = inpatient_before[['Internalpatientid', 
       'num_unique_units', 'num_transfers',
       'unique_admitting_specialty', 'unique_discharging_specialty',
       'DOMICILIARY', 'MEDICINE', 'NHCU', 'NON-COUNT', 'OTHERS', 'PSYCHIATRY',
       'SURGERY', 'Medical', 'Mental',
       'Others_Specialty', 'Rehab', 'Gerontology', 'Age at death']]
    
    tidy_dataset = tidy_dataset.merge(right = death, how="left", on="Internalpatientid")
    tidy_dataset = tidy_dataset[~tidy_dataset["Age at death"].isna()]

    tidy_dataset = tidy_dataset.merge(right = inpatient_before, how="left", on="Internalpatientid")

    tidy_dataset = tidy_dataset.drop(columns=["Age at death_y"])

    if datatype == "train": 
        tidy_dataset.to_csv('/home/hassan/lily/MLA/FDA/inpatient_cvd_mortality.csv')
        return '/home/hassan/lily/MLA/FDA/inpatient_cvd_mortality.csv'
    if datatype == "test": 
        tidy_dataset.to_csv('/home/hassan/lily/MLA/FDA/inpatient_cvd_mortality_test.csv')
        return '/home/hassan/lily/MLA/FDA/inpatient_cvd_mortality_test.csv'



def preprocess_outpatient_allcause_mortality(datatype):
    '''
    datatype = ["train", "test"]
    return: datapath
    train: /home/hassan/lily/MLA/FDA/outpatient_mortality.csv
    test: /home/hassan/lily/MLA/FDA/outpatient_mortality_test.csv

    Preprocess the dataset for target: all cause mortality
    '''
    if datatype == "train": outpatient_visits = pd.read_csv('/home/bhatti/dataset/VCHAMPS/outpatient_visits_train.csv', index_col=0)
    if datatype == "test": outpatient_visits = pd.read_csv('/data/public/MLA/VCHAMPS-Test/outpatient_visits_test.csv', index_col=0)

    outpatient_visits = outpatient_visits.drop([ 'Combatflag', 'Ionizingradiationflag', 'Serviceconnectedflag', 'Swasiaconditionsflag', 'Agentorangeflag'], axis=1)

    # Age at visit
    outpatient_visits['Age 00-20'] = [0] * len(outpatient_visits)
    outpatient_visits['Age 20-40'] = [0] * len(outpatient_visits)
    outpatient_visits['Age 40-60'] = [0] * len(outpatient_visits)
    outpatient_visits['Age 60-80'] = [0] * len(outpatient_visits)
    outpatient_visits['Age 80-100'] = [0] * len(outpatient_visits)
    outpatient_visits['Age 100-120'] = [0] * len(outpatient_visits)
    def age_category(data):
        data.loc[(data['Age at visit'] <= 20), 'Age 00-20'] = 1
        data.loc[(data['Age at visit'] > 20) & (data['Age at visit'] <= 40), 'Age 20-40'] = 1
        data.loc[(data['Age at visit'] > 40) & (data['Age at visit'] <= 60), 'Age 40-60'] = 1
        data.loc[(data['Age at visit'] > 60) & (data['Age at visit'] <= 80), 'Age 60-80'] = 1
        data.loc[(data['Age at visit'] > 80) & (data['Age at visit'] <= 100), 'Age 80-100'] = 1
        data.loc[(data['Age at visit'] > 100) & (data['Age at visit'] <= 120), 'Age 100-120'] = 1
        return data
    outpatient_visits = age_category(outpatient_visits)
    outpatient_visits
    
    # Cardiovascular

    outpatient_visits["CVD_outpatient"] = [0] * len(outpatient_visits)
    def cd_diagnosis(data):
        data.loc[(data["Second listed diagnosis icd10 subcategory"].str.contains('|'.join(cardiovascular))
        | data["Second listed diagnosis icd10 subcategory"].str.contains("Systolic (congestive) heart failure", regex=False)
        | data["Second listed diagnosis icd10 subcategory"].str.contains("Diastolic (congestive) heart failure", regex=False)
        | data["Second listed diagnosis icd10 subcategory"].str.contains("Combined systolic (congestive) and diastolic (congestive) heart failure", regex=False)
        | data["Second listed diagnosis icd10 subcategory"].str.contains("ST elevation (STEMI) myocardial infarction", regex=False)
        | data["Second listed diagnosis icd10 subcategory"].str.contains("Non-ST elevation (NSTEMI) myocardial infarction", regex=False)
        | data["First listed diagnosis icd10 subcategory"].str.contains('|'.join(cardiovascular))
        | data["First listed diagnosis icd10 subcategory"].str.contains("Systolic (congestive) heart failure", regex=False)
        | data["First listed diagnosis icd10 subcategory"].str.contains("Diastolic (congestive) heart failure", regex=False)
        | data["First listed diagnosis icd10 subcategory"].str.contains("Combined systolic (congestive) and diastolic (congestive) heart failure", regex=False)
        | data["First listed diagnosis icd10 subcategory"].str.contains("ST elevation (STEMI) myocardial infarction", regex=False)
        | data["First listed diagnosis icd10 subcategory"].str.contains("Non-ST elevation (NSTEMI) myocardial infarction", regex=False)), "CVD_outpatient"] = 1
        return data
    outpatient_visits = cd_diagnosis(outpatient_visits)

    outpatient_visits = outpatient_visits.drop(['First listed diagnosis icd10 subcategory', 'Second listed diagnosis icd10 subcategory'], axis=1)

    # #### Filter out visits that occur after death
    if datatype == "train": death = pd.read_csv("/home/bhatti/dataset/VCHAMPS/death_train.csv", index_col=0)
    if datatype == "test": death = pd.read_csv("/data/public/MLA/VCHAMPS-Test/death_test.csv", index_col=0)

    outpatient_visits = outpatient_visits.merge(death, how="left",on="Internalpatientid")
    outpatient_visits['Visit start date'] = pd.to_datetime(outpatient_visits['Visit start date'])
    outpatient_visits['Death date'] = pd.to_datetime(outpatient_visits['Death date'])
    outpatient_visits = outpatient_visits[~(outpatient_visits["Visit start date"] > outpatient_visits['Death date'])]

    outpatient_visits["Died"] = [1] * len(outpatient_visits)
    outpatient_visits.loc[outpatient_visits["Death date"].isna(), "Died"] = 0

    outpatient_visits['Visit start date'] = pd.to_datetime(outpatient_visits['Visit start date'])
    outpatient_visits['Death date'] = pd.to_datetime(outpatient_visits['Death date'])
    outpatient_visits["away_from_death"] = (outpatient_visits['Death date'] - outpatient_visits['Visit start date']).dt.days

    # ### Final dataset

    tidy_dataset = []
    for ids, group in tqdm(outpatient_visits.groupby("Internalpatientid")):
        # group = group.sort_values(by = ['Visit start date'],ascending=True).reset_index(drop = True)
        num_visits = group["Age at visit"].nunique()

        age_mean = group["Age at visit"].mean()
        age_std = group["Age at visit"].std()
        if group["Age at visit"].nunique() == 1: age_std = 0

        min_age = group["Age at visit"].min()
        max_age = group["Age at visit"].max()
        freq = len(group)/(math.floor(max_age - min_age) + 1)

        # num_cvd_visits = group['CVD_outpatient'].sum()
        if  group['CVD_outpatient'].sum() > 0: CVD = 1
        else: CVD = 0

        last_visit_date = group["Visit start date"].max()
        away_from_death = group["away_from_death"].min()

        if group["Died"].sum() > 0: died = 1
        else: died = 0

        df = pd.DataFrame(data={'Internalpatientid': [ids], 'num_visits': [len(group)], 
                                "CVD": [CVD], "last_visit_date": [last_visit_date], "Age 00-20": group["Age 00-20"].sum(),
                                'Age 20-40': group["Age 20-40"].sum(), 'Age 40-60': group["Age 40-60"].sum(), 'Age 60-80':group["Age 60-80"].sum(), 
                                'Age 80-100':group["Age 80-100"].sum(), 'Age 100-120':group["Age 100-120"].sum(), 
                                "age_mean": [age_mean], "age_std": [age_std], "freq": [round(freq,2)], "Died": [died], "away_from_death": [away_from_death]
                                })
        
        df = df.reset_index(drop=True)
        tidy_dataset.append(df)
        
    tidy_dataset = pd.concat(tidy_dataset)

    tidy_dataset['died_within_125days'] = [0] * len(tidy_dataset)

    def age_category(data):
        data.loc[(data['away_from_death'] < 125), 'died_within_125days'] = 1
        return data
    tidy_dataset = age_category(tidy_dataset)

    if datatype == "train": 
        tidy_dataset.to_csv("/home/hassan/lily/MLA/FDA/outpatient_mortality.csv")
        return '/home/hassan/lily/MLA/FDA/outpatient_mortality.csv'
    if datatype == "test": 
        tidy_dataset.to_csv("/home/hassan/lily/MLA/FDA/outpatient_mortality_test.csv")
        return "/home/hassan/lily/MLA/FDA/outpatient_mortality_test.csv"



def preprocess_lab_results_cvd_mortality(datatype):
    return


def preprocess_lab_results_readmission(datatype):
    return


def preprocess_lab_results_allcause_mortality(datatype):
    return
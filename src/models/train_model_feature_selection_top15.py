import pandas as pd
import numpy as np
from importlib import reload
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.validation import column_or_1d
import pickle
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import argparse
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from transformation import *    
from grid_search_cv_f1 import *
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics   #Additional scklearn functions
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier


dict_data = {
    'readmission': '/home/daisy/FDA_Dataset/inpatient_all_final_1.csv', 
    'readmission_cvd': '/home/daisy/FDA_Dataset/inpatient_CVD_final_1.csv',
    "mortality": '/home/daisy/FDA_Dataset/final_allcause_mortality_train_1.csv',
    "mortality_cvd": '/home/daisy/FDA_Dataset/final_cvd_mortality_train_1.csv'
}

def prepare_dataset(target):
    # Import Data
    path =  dict_data[target]
    data = pd.read_csv(path).iloc[:,1:]
    
    readmission_features =['Age 100-120 hypertension',
                            'Age 20-40 healthy',
                            'Age 40-60',
                            'Age 40-60 hypertension',
                            'Age 60-80 healthy',
                            'Age 60-80 hypertension',
                            'Age 60-80 hypotension',
                            'Age 80-100 healthy',
                            'Age 80-100 hypertension',
                            'Age 80-100 hypotension',
                            'CVD',
                            'DOMICILIARY',
                            'Ethnicity_0',
                            'Height',
                            'Num med per admission mean',
                            'Others_Specialty',
                            'Pulse oximetry max',
                            'Pulse oximetry mean',
                            'Pulse oximetry min',
                            'Pulse oximetry std',
                            'Races_0',
                            'Races_2',
                            'SURGERY',
                            'Total medications',
                            'Weight',
                            'age_std',
                            'lab_age_mean',
                            'lab_age_std',
                            'lab_count',
                            'lab_freq',
                            'num_stays',
                            'period mean',
                            'period std',
                            'specialty count',
                            'specialty medical count',
                            'stay_length',
                            'stay_mean',
                            'stay_std',
                            'total_procedure','Ethnicity', 'Gender', 'Races', 'Ethnicity_1', 'Ethnicity_2', 'Races_1','Races_2', 'Races_3', 'Ruca category encoded']
    mortality_cvd_features = ['Age 00-20',
                                'Age 100-120 hypertension',
                                'Age 20-40 healthy',
                                'Age 40-60 hypertension',
                                'Age 60-80 hypertension',
                                'Age 60-80 hypotension',
                                'Age 80-100 healthy',
                                'Age 80-100 hypertension',
                                'Age 80-100 hypotension',
                                'Ethnicity_0',
                                'Ethnicity_2',
                                'Gender',
                                'Height',
                                'Pulse oximetry max',
                                'Pulse oximetry mean',
                                'Pulse oximetry min',
                                'Pulse oximetry std',
                                'Races_0',
                                'Races_3',
                                'Ruca category encoded',
                                'Total medications',
                                'Weight',
                                'age_min',
                                'age_std',
                                'lab_age_mean',
                                'lab_age_std',
                                'lab_count',
                                'lab_freq',
                                'num_cvd_admission',
                                'num_visits',
                                'period mean',
                                'period std',
                                'stay_length',
                                'stay_max',
                                'stay_mean',
                                'stay_min',
                                'total_procedure','Ethnicity', 'Gender', 'Races', 'Ethnicity_1', 'Ethnicity_2', 'Races_1','Races_2', 'Races_3', 'Ruca category encoded']
    mortality_features = ['Age 00-20',
                        'Age 100-120 hypertension',
                        'Age 20-40',
                        'Age 20-40 healthy',
                        'Age 40-60',
                        'Age 40-60 healthy',
                        'Age 40-60 hypertension',
                        'Age 60-80',
                        'Age 60-80 hypotension',
                        'Age 80-100',
                        'Age 80-100 healthy',
                        'Age 80-100 hypertension',
                        'Age 80-100 hypotension',
                        'CVD',
                        'Ethnicity_0',
                        'Ethnicity_2',
                        'Gender',
                        'Height',
                        'Pulse oximetry max',
                        'Pulse oximetry min',
                        'Races_0',
                        'Races_3',
                        'Ruca category encoded',
                        'Total medications',
                        'Weight',
                        'age_mean',
                        'age_std',
                        'freq',
                        'lab_age_mean',
                        'lab_age_std',
                        'lab_count',
                        'lab_freq',
                        'mean age at specailty',
                        'num_immunization',
                        'num_visits',
                        'period mean',
                        'period std',
                        'total_procedure','Ethnicity', 'Gender', 'Races', 'Ethnicity_1', 'Ethnicity_2', 'Races_1','Races_2', 'Races_3', 'Ruca category encoded']

    if target == "readmission":
        #X = data.drop(columns = ['Internalpatientid', 'CVD_readmission', 'readmission within 300 days'])
        X = data[readmission_features]
        y = column_or_1d(data[['readmission within 300 days']])
    elif target == "readmission_cvd":
        # X = data.drop(columns = ['Internalpatientid', 'CVD_readmission', 'readmission within 300 days'])
        X = data[readmission_features]
        y = column_or_1d(data[['CVD_readmission']])
    elif target == "mortality":
        #X = data.drop(columns = ['Internalpatientid', 'died_within_125days'])
        drops = set(data.columns).difference(set(mortality_features))
        X = data.drop(columns = drops)
        y = column_or_1d(data[['died_within_125days']])
    elif target == 'mortality_cvd':
        drops = set(data.columns).difference(set(mortality_cvd_features))
        X = data.drop(columns = drops)
        y = column_or_1d(data[['died_by_cvd']])
        

    # # Split Train and Test (?? 似乎不用)
    # X_train_ad1, X_test_ad1, y_train_ad1, y_test_ad1 = train_test_split(X_admission1, Y_admission1, test_size=0.20, random_state=42)
    # Transform Data
    transform_steps = [("ImputeNumeric", ImputeNumeric()),
                ('RemoveSkewnessKurtosis', RemoveSkewnessKurtosis()),
                ('StandardizeStandardScaler', Standardize(RobustScaler()))]
    transform_pipeline = Pipeline(transform_steps)
    X = transform_pipeline.transform(X)
    unique, counts = np.unique(y, return_counts=True)
    print(unique, counts)
    X.fillna(0,inplace=True)

    # Balance the dataset
    sme = SMOTEENN(random_state=42)
    X, y = sme.fit_resample(X, y)
    unique, counts = np.unique(y, return_counts=True)
    print(unique, counts)
    return X,y


def train_model(X,y,model_type):
    if model_type=='LogisticRegression':
        gsearch = LogisticRegression_Grid_CV(X,y,LogisticRegression_param)
        print(gsearch.best_score_)
        return LogisticRegression(**gsearch.best_params_).fit(X,y)

    elif model_type=='LinearDiscriminant':
        gsearch = LinearDiscriminant_Grid_CV(X,y,LinearDiscriminant_param)
        print(gsearch.best_score_)
        return LinearDiscriminantAnalysis(**gsearch.best_params_).fit(X,y)

    elif model_type=='DecisionTree':
        gsearch = DecisionTree_Grid_CV(X,y,DecisionTree_param)
        print(gsearch.best_score_)
        return DecisionTreeClassifier(**gsearch.best_params_).fit(X,y)

    elif model_type=='RandomForest':
        gsearch = RandomForest_Grid_CV(X,y,RandomForest_param)
        print(gsearch.best_score_)
        return RandomForestClassifier(**gsearch.best_params_).fit(X,y)

    elif model_type=='XGBoost':
        gsearch = XGBoost_Grid_CV(X,y,XGBoost_param)
        print(gsearch.best_score_)
        return xgb.XGBClassifier(**gsearch.best_params_).fit(X,y)

    elif model_type=='AdaBoost':
        gsearch = AdaBoost_Grid_CV(X,y,AdaBoost_param)
        print(gsearch.best_score_)
        return gsearch.best_estimator_.fit(X,y)

    elif model_type=='LGBM':
        gsearch = LGBM_Grid_CV(X,y,LGBM_param)
        print(gsearch.best_score_)
        return LGBMClassifier(**gsearch.best_params_).fit(X,y)

    else: raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Ideally target should be readmission, readmission_cvd, motality, motality_cvd
    parser.add_argument("--model_type", help="select model architecture", type=str)
    parser.add_argument("--target", help="select target", type=str)
    
    args = parser.parse_args()

    print(f"Selected model type is: {args.model_type}")
    print(f"Target: {args.target}")

    X, y = prepare_dataset(args.target)

    clf = train_model(X,y,args.model_type)

    filename = f"/home/lily/FDA/models/{args.model_type}_{args.target}_feature_selection.sav"
    pickle.dump(clf, open(filename, 'wb'))
  

    
    




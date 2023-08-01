#!/bin/bash

## tuning for min max filtered!
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality" --model_type "LogisticRegression" >/home/daisy/FDA/src/logs/LogisticRegression_mortality.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission" --model_type "LogisticRegression" >/home/daisy/FDA/src/logs/LogisticRegression_readmission.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission_cvd" --model_type "LogisticRegression" >/home/daisy/FDA/src/logs/LogisticRegression_readmission_cvd.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality_cvd" --model_type "LogisticRegression" >/home/daisy/FDA/src/logs/LogisticRegression_mortality_cvd.py.log 2>&1

python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission" --model_type "LinearDiscriminant" >/home/daisy/FDA/src/logs/LinearDiscriminant_readmission.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality" --model_type "LinearDiscriminant" >/home/daisy/FDA/src/logs/LinearDiscriminant_mortality.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission_cvd" --model_type "LinearDiscriminant" >/home/daisy/FDA/src/logs/LinearDiscriminant_readmission_cvd.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality_cvd" --model_type "LinearDiscriminant" >/home/daisy/FDA/src/logs/LinearDiscriminant_mortality_cvd.py.log 2>&1

python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission" --model_type "DecisionTree" >/home/daisy/FDA/src/logs/DecisionTree_readmission.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality" --model_type "DecisionTree" >/home/daisy/FDA/src/logs/DecisionTree_mortality.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission_cvd" --model_type "DecisionTree" >/home/daisy/FDA/src/logs/DecisionTree_readmission_cvd.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality_cvd" --model_type "DecisionTree" >/home/daisy/FDA/src/logs/DecisionTreen_mortality_cvd.py.log 2>&1


python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission" --model_type "RandomForest" >/home/daisy/FDA/src/logs/RandomForest_readmission.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality" --model_type "RandomForest" >/home/daisy/FDA/src/logs/RandomForest_mortality.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission_cvd" --model_type "RandomForest" >/home/daisy/FDA/src/logs/RandomForest_readmission_cvd.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality_cvd" --model_type "RandomForest" >/home/daisy/FDA/src/logs/RandomForest_mortality_cvd.py.log 2>&1


python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission" --model_type "XGBoost" >/home/daisy/FDA/src/logs/XGBoost_readmission.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality" --model_type "XGBoost" >/home/daisy/FDA/src/logs/XGBoost_mortality.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission_cvd" --model_type "XGBoost" >/home/daisy/FDA/src/logs/XGBoost_readmission_cvd.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality_cvd" --model_type "XGBoost" >/home/daisy/FDA/src/logs/XGBoost_mortality_cvd.py.log 2>&1

python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission" --model_type "AdaBoost" >/home/daisy/FDA/src/logs/AdaBoost_readmission.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality" --model_type "AdaBoost" >/home/daisy/FDA/src/logs/AdaBoost_mortality.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission_cvd" --model_type "AdaBoost" >/home/daisy/FDA/src/logs/AdaBoost_readmission_cvd.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality_cvd" --model_type "AdaBoost" >/home/daisy/FDA/src/logs/AdaBoost_mortality_cvd.py.log 2>&1

python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission" --model_type "LGBM" >/home/daisy/FDA/src/logs/LGBM_readmission.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality" --model_type "LGBM" >/home/daisy/FDA/src/logs/LGBM_mortality.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "readmission_cvd" --model_type "LGBM" >/home/daisy/FDA/src/logs/LGBM_readmission_cvd.py.log 2>&1
python /home/daisy/FDA/src/models/train_model_feature_selection.py --target "mortality_cvd" --model_type "LGBM" >/home/daisy/FDA/src/logs/LGBM_mortality_cvd.py.log 2>&1
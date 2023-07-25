#!/bin/bash

## tuning for min max filtered!
python /home/vivi/FDA/src/models/train_model.py --target "readmission" --model_type "LogisticRegression" >/home/vivi/FDA/src/logs/LogisticRegression_readmission.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "readmission" --model_type "LinearDiscriminant" >/home/vivi/FDA/src/logs/LinearDiscriminant_readmission.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "readmission" --model_type "DecisionTree" >/home/vivi/FDA/src/logs/DecisionTree_readmission.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "readmission" --model_type "RandomForest" >/home/vivi/FDA/src/logs/RandomForest_readmission.py.log 2>&1

python /home/vivi/FDA/src/models/train_model.py --target "readmission_cvd" --model_type "LogisticRegression" >/home/vivi/FDA/src/logs/LogisticRegression_readmission_cvd.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "readmission_cvd" --model_type "LinearDiscriminant" >/home/vivi/FDA/src/logs/LinearDiscriminant_readmission_cvd.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "readmission_cvd" --model_type "DecisionTree" >/home/vivi/FDA/src/logs/DecisionTree_readmission_cvd.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "readmission_cvd" --model_type "RandomForest" >/home/vivi/FDA/src/logs/RandomForest_readmission_cvd.py.log 2>&1

# python /home/vivi/FDA/src/models/train_model.py --target "mortality" --model_type "LogisticRegression" >/home/vivi/FDA/src/logs/LogisticRegression_mortality.py.log 2>&1
# python /home/vivi/FDA/src/models/train_model.py --target "mortality" --model_type "LinearDiscriminant" >/home/vivi/FDA/src/logs/LinearDiscriminant_mortality.py.log 2>&1
# python /home/vivi/FDA/src/models/train_model.py --target "mortality" --model_type "DecisionTree" >/home/vivi/FDA/src/logs/DecisionTree_mortality.py.log 2>&1
# python /home/vivi/FDA/src/models/train_model.py --target "mortality" --model_type "RandomForest" >/home/vivi/FDA/src/logs/RandomForest_mortality.py.log 2>&1

python /home/vivi/FDA/src/models/train_model.py --target "mortality_cvd" --model_type "LogisticRegression" >/home/vivi/FDA/src/logs/LogisticRegression_mortality_cvd.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "mortality_cvd" --model_type "LinearDiscriminant" >/home/vivi/FDA/src/logs/LinearDiscriminant_mortality_cvd.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "mortality_cvd" --model_type "DecisionTree" >/home/vivi/FDA/src/logs/DecisionTreen_mortality_cvd.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "mortality__cvd" --model_type "RandomForest" >/home/vivi/FDA/src/logs/RandomForest_mortality_cvd.py.log 2>&1

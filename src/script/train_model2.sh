#!/bin/bash

python /home/vivi/FDA/src/models/train_model.py --target "mortality_cvd" --model_type "LogisticRegression" >/home/vivi/FDA/src/logs/LogisticRegression_mortality_cvd.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "mortality_cvd" --model_type "LinearDiscriminant" >/home/vivi/FDA/src/logs/LinearDiscriminant_mortality_cvd.py.log 2>&1
python /home/vivi/FDA/src/models/train_model.py --target "mortality_cvd" --model_type "DecisionTree" >/home/vivi/FDA/src/logs/DecisionTreen_mortality_cvd.py.log 2>&1


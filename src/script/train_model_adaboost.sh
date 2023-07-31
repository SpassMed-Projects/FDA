#!/bin/bash

python /home/lily/FDA/src/models/train_model.py --target "mortality_cvd" --model_type "AdaBoost" >/home/lily/FDA/src/logs/AdaBoost_mortality_cvd.py.log 2>&1
python /home/lily/FDA/src/models/train_model.py --target "mortality" --model_type "AdaBoost" >/home/lily/FDA/src/logs/AdaBoost_mortality.py.log 2>&1
python /home/lily/FDA/src/models/train_model.py --target "readmission" --model_type "AdaBoost" >/home/lily/FDA/src/logs/AdaBoost_readmission.py.log 2>&1
python /home/lily/FDA/src/models/train_model.py --target "readmission_cvd" --model_type "AdaBoost" >/home/lily/FDA/src/logs/AdaBoost_readmission_cvd.py.log 2>&1
#!/bin/bash

## tuning for min max filtered!
python ../models/train_model.py --target "readmission" --model_type "LogisticRegression" >../logs/LogisticRegression_readmission.py.log 2>&1

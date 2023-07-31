from sklearn.ensemble import VotingClassifier
import argparse
import pickle

#If ‘hard’, uses predicted class labels for majority rule voting. 
# Else if ‘soft’, predicts the class label based on the argmax of the 
# sums of the predicted probabilities,
# which is recommended for an ensemble of well-calibrated classifiers.
LR_best = pickle.load(open('/home/daisy/FDA/models/LogisticRegression_readmission.sav','rb'))
RF_best = pickle.load(open('/home/daisy/FDA/models/RandomForest_readmission.sav','rb'))
DT_best = pickle.load(open('/home/daisy/FDA/models/DecisionTree_readmission.sav','rb'))
def train_ensemble(X,y):
        eclf1 = VotingClassifier(estimators=[('rf', RF_best), ('lr_readmission', LR_best),
        ('dt', DT_best)], voting='soft', n_jobs=3)
        eclf1 = eclf1.fit(X, y)

def train_ensemble(X,y):
        eclf1 = VotingClassifier(estimators=[(model_name1, model1), (model_name2, model2),
        (model_name3, model3)], voting='soft', n_jobs=3)
        eclf1 = eclf1.fit(X, y)

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('file1', type=argparse.FileType('r'))
        parser.add_argument('file2', type=argparse.FileType('r'))
        parser.add_argument('file3', type=argparse.FileType('r'))
        parser.add_argument("--model_name1", help="select model name1", type=str)
        parser.add_argument("--model_name2", help="select model name2", type=str)
        parser.add_argument("--model_name3", help="select model name3", type=str)
        args = parser.parse_args()

        model_name1 = args.model_name1
        model_name2 = args.model_name2
        model_name3 = args.model_name3

        model1 = pickle.load(open(args.file1,'rb'))
        model2 = pickle.load(open(args.file2,'rb'))
        model3 = pickle.load(open(args.file3,'rb'))

'''
Create a file: best_models
RF_best, gsearch_RF # get best_estimator from saved model
LR_best, gsearch_LR
DT_best, gsearch_LR

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()
with open(args.filename) as file:
'''

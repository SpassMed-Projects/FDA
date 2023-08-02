import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.validation import column_or_1d
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import train_model
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
from grid_search_cv import *
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
import test_results


statistics_metrics = pd.DataFrame(['Area under the precision recall curve (AUPRC)',
                                       'Area under the Receiver Operating Characteristic (AUROC)',
                                       'Overall Accuracy',
                                       'Sum of Sensitivity and Specificity',
                                       'Sensitivity',
                                       'Specificity',
                                       'Precision',
                                       'Negative Predictive Value',
                                       'Positive Likelihood Ratio',
                                       'Negative Likelihood Ratio',
                                       'F1 score'], columns=['statistics_metrics'])

def ensemble(target, voting):

    model_names = [ "LogisticRegression", "XGBoost", "DecisionTree","LinearDiscriminant", "LGBM", "AdaBoost"]
    

    X_train, y_train = train_model.prepare_dataset(target)
    results = pd.read_csv("/home/vivi/FDA/reports/test_statistics_metrics_feature_selection_all_lily.csv", index_col=0)
    models = []
    weights = []
    for m in model_names:
        model_name = f"/home/lily/FDA/models/{m}_{target}_3.sav"
        clf = pickle.load(open(model_name,'rb'))
        result_name = f"{m}_{target}_3"
    
        # if m =="LGBM": 

        #     X_test, y_test = test_results.prepare_dataset(target, clf.feature_name_)

        # else: X_test, y_test = test_results.prepare_dataset(target, clf.feature_names_in_)
        # predict_label, predict_contin = test_results.make_prediction(X_test,target,clf)
        weights.append(results.iloc[-1][result_name])
        models.append((m, clf))

    if voting == "hard":
        eclf = VotingClassifier(estimators=models, voting='hard', weights=weights)
        eclf.fit(X_train,y_train)

        filename = f"/home/lily/FDA/models/{target}_{voting}_ensemble.sav"
        pickle.dump(clf, open(filename, 'wb'))

        X_test, y_test = test_results.prepare_dataset(target, eclf.feature_names_in_)
        predict_label = eclf.predict(X_test)
    else: 
        eclf = VotingClassifier(estimators=models, voting='soft')
        eclf.fit(X_train,y_train)

        filename = f"/home/lily/FDA/models/{target}_{voting}_ensemble.sav"
        pickle.dump(clf, open(filename, 'wb'))
        
        X_test, y_test = test_results.prepare_dataset(target, eclf.feature_names_in_)
        predict_label, predict_contin = test_results.make_prediction(X_test,target,eclf)
    
    return test_results.calculate_score(y_test, predict_label)


if __name__ == '__main__':
    
    targets = ["mortality", "mortality_cvd", "readmission", "readmission_cvd"]

    for v in ["soft", "hard"]:
        for target in targets:
            score = ensemble(target, v)
            statistics_metrics[target] = score
        statistics_metrics.to_csv(f"/home/lily/FDA/reports/{v}_ensembled_test_statistics_metrics.csv")
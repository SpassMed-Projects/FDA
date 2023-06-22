''' 
Classic machine learning models include:
- Descision Tree sklearn.tree.DecisionTreeClassifier
- Logistic Regression 
- XGB
- LGBM
- Random Forest
- SVM
- LDA
- Gaussion Naive Bayes
- MLP
Imported from SKLearn library

'''

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  
from sklearn.model_selection import KFold

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

classifiers = [SGDClassifier(), \
        GaussianNB(), \
        RandomForestClassifier(max_depth=5,n_estimators=10), \
        MLPClassifier(alpha=0.05), \
        AdaBoostClassifier()]




"""

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    # Section 3.1
    # 1. SGDClassifier: support vector machine with a linear kernel.
    # 2. GaussianNB: a Gaussian naive Bayes classifier.
    # 3. RandomForestClassifier: with a maximum depth of 5, and 10 estimators.
    # 4. MLPClassifier: A feed-forward neural network, with α = 0.05.
    # 5. AdaBoostClassifier: with the default hyper-parameters.

    classifier_names = ["SGDClassifier", "GaussianNB", "RandomForestClassifier", \
        "MLPClassifier", "AdaBoostClassifier"]

    accuracies = []
    recalls = []
    precisions = []
    matrice = []
    maxAcc = 0
    iBest = 0

    for i in range(5):
        mdl = classifiers[i]
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        matrice.append(confusion_matrix(y_test, y_pred))
        accuracies.append(accuracy(matrice[i]))
        if accuracies[i] > maxAcc:
            maxAcc = accuracies[i]
            iBest = i
        recalls.append(recall(matrice[i]))
        precisions.append(precision(matrice[i]))
    
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        #     outf.write(f'Results for {classifier_name}:\n')  # Classifier name
        #     outf.write(f'\tAccuracy: {acc:.4f}\n')
        #     outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
        #     outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
        #     outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

        for classifier_name, acc, recl, preci,conf_matrix in \
            zip(classifier_names, accuracies, recalls, precisions, matrice):
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recl]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in preci]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    
    clf = classifiers[iBest]
    sample_sizes = [1000, 5000, 10000, 15000, 20000]
    accuracies = []
    for i in sample_sizes:
        random_indices = np.random.choice(len(X_train), i, replace=False)
        X_sample = X_train[random_indices]
        y_sample = y_train[random_indices]
        if i == 1000:
            X_1k, y_1k = X_sample, y_sample
        mdl = clf.fit(X_sample,y_sample)
        y_pred = mdl.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)
        accuracies.append(accuracy(matrix))
    
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {acc:.4f}\n'))
        for num_train, acc in zip(sample_sizes, accuracies):
             outf.write(f'{num_train}: {acc:.4f}\n')

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    # Section 3.3
    k_feats = [5, 50]
    p_values_list = []
    for k_size in k_feats:
        selector = SelectKBest(k=k_size)
        X_new = selector.fit_transform(X_train, y_train)
        selector.fit(X_new,y_train)
        p_values_list.append(selector.pvalues_)
    
    datas = [[X_1k, y_1k],[X_train, y_train]]
    accuracies = []
    clf = classifiers[i]
    matrice = []
    datas_transform = []
    filters = []
    for data in datas:
        selector = SelectKBest(k=5)
        X_new = selector.fit_transform(data[0], data[1])
        filters.append(selector.get_support(indices=True))
        datas_transform.append(X_new)
        mdl = clf.fit(X_new, data[1])
        y_pred = mdl.predict(selector.transform(X_test))
        matrice.append(confusion_matrix(y_test, y_pred))
        accuracies.append(accuracy(matrice[-1]))
    
    feature_intersection = set(filters[0]).intersection(set(filters[1]))
    top_5 = set(filters[1])
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
    #     # Prepare the variables with corresponding names, then uncomment
    #     # this, so it writes them to outf.
        
    #     # for each number of features k_feat, write the p-values for
    #     # that number of features:
    #         # outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
        for k_feat, p_values in zip(k_feats, p_values_list):
            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
        
        # outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        # outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        # outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        # outf.write(f'Top-5 at higher: {top_5}\n')
        outf.write(f'Accuracy for 1k: {accuracies[0]:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracies[1]:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')

def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    # Section 3.4  
    kfs = KFold(n_splits = 5, shuffle = True)
    X = np.concatenate((X_train,X_test))
    y = np.concatenate((y_train,y_test))

    indice = kfs.split(X)
    folds = []
 
    for train_index , test_index in indice:
        X_train , X_test = X[train_index],X[test_index]
        y_train , y_test = y[train_index], y[test_index]
        accuracies = []
        for clf in classifiers:
            mdl = clf.fit(X_train,y_train)
            y_pred = mdl.predict(X_test)
            matrix = confusion_matrix(y_test, y_pred)
            accuracies.append(accuracy(matrix))
        folds.append(accuracies)

    clfs_accuracies = np.transpose(folds)
    p_values = []
    for index in range(5):
        if index != i:
            p_values.append(ttest_rel(clfs_accuracies[index], clfs_accuracies[i]))

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
        for kfold_accuracies in folds:
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.
    output_dir = args.output_dir
    data = np.load(args.input)['arr_0']
    X = np.array([np.array(i[:-1]) for i in data])
    y = np.array([i[-1] for i in data])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    iBest = class31(output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(output_dir, X_train, X_test, y_train, y_test, iBest)

"""
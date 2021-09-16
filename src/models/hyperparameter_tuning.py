# Internal
from src.models.initialize_models import initialize_model, construct_kwargs
from src.train.k_fold_cross_validation import k_fold_prediction
from src.evaluation.accuracy import accuracy
from src.read_write import to_csv
# External
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

def grid_search_helper(m, hyperparameters, features, data, OUTPUT_FILE):
    try:
        kwargs = construct_kwargs(hyperparameters, features)
        model = initialize_model(m, kwargs)
        cnf_matrix = k_fold_prediction(model, data)
        model_accuracy = accuracy(model, cnf_matrix)

        print(model_accuracy)
        to_csv(OUTPUT_FILE, model_accuracy.csv_str())
    except (ValueError):
        return

def grid_search_tree(OUTPUT_FILE, data):
    criterion = ['gini', 'entropy']
    max_depth = [3, 4, 5]
    max_features = [55, 56]
    max_leaf_nodes = list(range(2, 30))
    min_impurity_decrease = [0, 0.1]
    min_samples_leaf = [2, 3]

    hyperparameters = ["criterion", "max_depth", "max_features", "max_leaf_nodes", "min_impurity_decrease", "min_samples_leaf"]
    
    for c in criterion:
        for d in max_depth:
            for f in max_features:
                for l in max_leaf_nodes:
                    for i in min_impurity_decrease:
                        for ml in min_samples_leaf:
                            features = [c,d,f,l,i,ml]
                            grid_search_helper(DecisionTreeClassifier, hyperparameters, features, data, OUTPUT_FILE)
                            
def grid_search_logreg(OUTPUT_FILE, data):
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    penalty = ['l1', 'l2', 'elasticnet', 'none']
    solver = ['saga', 'sag', 'liblinear', 'lbfgs', 'newton-cg']

    hyperparameters = ["C", "penalty", "solver"]

    for c in C:
        for p in penalty:
            for s in solver:
                features = [c,p,s]
                grid_search_helper(LogisticRegression, hyperparameters, features, data, OUTPUT_FILE)

def grid_search_ridge(OUTPUT_FILE, data):
    alpha = [0.0001, 0.001, 0.01, 0.1, 1, 2, 4, 10]
    solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

    hyperparameters = ["alpha", "solver"]

    for a in alpha:
        for s in solver:
            features = [a,s]
            grid_search_helper(RidgeClassifier, hyperparameters, features, data, OUTPUT_FILE)

def grid_search_svm(OUTPUT_FILE, data):
    degree = [0, 1, 2, 3, 4, 5, 6]
    C = [0.0001, 100]
    gamma = [0.0001, 0.001, 0.01, 0.1, 1, 2, 4, 10, 100]
    kernel = ['rbf', 'linear']

    hyperparameters = ["degree", "C", "gamma", "kernel"]

    for d in degree:
        for c in C:
            for g in gamma:
                for k in kernel:
                    features = [d,c,g,k]
                    grid_search_helper(svm.SVC, hyperparameters, features, data, OUTPUT_FILE)

def grid_search_neighbors(OUTPUT_FILE, data):
    leaf_size = list(range(1, 10))
    n_neighbors = list(range(20, 40))
    p = [1, 2, 3]

    hyperparameters = ["leaf_size", "n_neighbors", "p"]

    for ls in leaf_size:
        for n in n_neighbors:
            for p_iter in p:
                features = [ls,n,p_iter]
                grid_search_helper(neighbors.KNeighborsClassifier, hyperparameters, features, data, OUTPUT_FILE)

def grid_search_naive(OUTPUT_FILE, data):
    var_smoothing = np.logspace(0, -9, num=100)

    hyperparameters = ["var_smoothing"]

    for vs in var_smoothing:
        features = [vs]
        grid_search_helper(GaussianNB, hyperparameters, features, data, OUTPUT_FILE)

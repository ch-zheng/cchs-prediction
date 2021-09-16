from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def initialize_model(func, kwargs):
    return func(**kwargs)

# def initialize_model(func):
#     return func()

def construct_kwargs(param_names, param_values):
    kwargs = dict(zip(param_names, param_values))
    return kwargs

# class models():
#     def __init__(self, **model_params):
#         self.tree = self.init_tree(model_params['tree']) if model_params['tree'] != None else self.init_tree(),
#         self.logreg = self.init_logreg(model_params['logreg']) if model_params['logreg'] != None else self.init_logreg(),
#         self.ridge = self.init_ridge(model_params['ridge']) if model_params['ridge'] != None else self.init_ridge(),
#         self.svm = self.init_svm(model_params['svm']) if model_params['svm'] != None else self.init_svm(),
#         self.neighbors = self.init_neighbors(model_params['neighbors']) if model_params['neighbors'] != None else self.init_neighbors(),
#         self.naive = self.init_naive(model_params['naive']) if model_params['naive'] != None else self.init_naive()
    
#     # Decision Tree
#     def init_tree(self, **tree_params):
#         self.models['tree'] = DecisionTreeClassifier(tree_params)

#     def init_tree(self):
#         return DecisionTreeClassifier()

#     # Logistic Regression
#     def init_logreg(self, **logreg_params):
#         return LogisticRegression(logreg_params)

#     def init_logreg(self):
#         return LogisticRegression()

#     # Ridge Regression
#     def init_ridge(self, **ridge_params):
#         return RidgeClassifier(ridge_params)

#     def init_ridge(self):
#         return RidgeClassifier()

#     # Support Vector Machine
#     def init_svm(self, **svm_params):
#         return svm.SVC(svm_params)

#     def init_svm(self):
#         return svm.SVC()

#     # K-Nearest Neighbors
#     def init_neighbors(self, **neighbors_params):
#         return neighbors.KNeighborsClassifier(neighbors_params)

#     def init_neighbors(self):
#         return neighbors.KNeighborsClassifier()

#     # Naive Bayes
#     def init_naive(self, **naive_params):
#         return GaussianNB(naive_params)

#     def init_naive(self):
#         return GaussianNB()

# def initialize_models(**model_params):
#     models = {}
    
#     tree = DecisionTreeClassifier()
#     logreg = LogisticRegression()
#     ridge = RidgeClassifier()
#     svm_model = svm.SVC()
#     knearest = neighbors.KNeighborsClassifier()
#     naive = GaussianNB()
#     #mlp = MLPClassifier()
    
#     models = {
#         "Decision Tree": tree,
#         "Logistic Regression": logreg,
#         "Ridge Regression": ridge,
#         "K-Nearest Neighbors": knearest,
#         "SVM": svm_model,
#         "Naive Bayes": naive
#         #"MLP": mlp
#     }
    
#     return models
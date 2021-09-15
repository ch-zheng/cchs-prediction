from model import SKModel
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Description: Gaussian Naive Bayes classifier

# ML model class zoo

class Ridge(SKModel):
    def __init__(self):
        self.model = RidgeClassifier(class_weight='balanced', solver='saga')
        self.test_params = {
            'alpha': np.logspace(-8, 8, num=17, base=2)
        }

class Logistic(SKModel):
    def __init__(self):
        self.model = LogisticRegression(
            class_weight='balanced',
            solver='saga',
            max_iter=10000,
            n_jobs=-1,
            multi_class='ovr')
        self.test_params = {
            'C': np.logspace(-8, 8, num=17, base=2)
        }

class SVM(SKModel):
    def __init__(self):
        self.model = LinearSVC(dual=False, class_weight='balanced', max_iter=10000)
        self.test_params = {
            'C': np.logspace(0, 9, num=10, base=2)
        }

class Neighbors(SKModel):
    def __init__(self):
        self.model = KNeighborsClassifier(n_jobs=-1)
        self.test_params = {
            'n_neighbors': list(range(1, 10)),
            'weights': ('uniform', 'distance'),
            'p': list(range(1, 4))
        }

class NaiveBayes(SKModel):
    def __init__(self):
        self.model = GaussianNB()
        self.test_params = {
            'var_smoothing': np.logspace(-16, 0, num=17)
        }

class DecisionTree(SKModel):
    def __init__(self):
        self.model = DecisionTreeClassifier(class_weight='balanced')
        self.test_params = {
            'criterion': ('gini', 'entropy'),
            'max_depth': list(range(2, 8)),
            'min_samples_split': list(range(2, 8)),
            'min_samples_leaf': list(range(1, 8)),
            'max_features': ('sqrt', 'log2', None)
        }

## Import models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifier
from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
# other libraries
import numpy as np
from sklearn.model_selection import cross_val_score
from pathlib import Path

## Load data
X = np.load('data/samples.npy')
y = np.load('data/labels.npy')

## Initialize models
tree = DecisionTreeClassifier()
logreg = LogisticRegression(max_iter=1000)
# tktk lasso (TODO)
# tktk polyreg (TODO)
ridge = RidgeClassifier()
knearest = neighbors.KNeighborsClassifier(n_jobs=-1)
svm_model = svm.SVC()
naive = GaussianNB()

# add lasso, polyreg to dictionary (TODO)
models = {
    "Decision Tree": tree,
    "Logistic Regression": logreg,
    "Ridge Regression": ridge,
    "K-Nearest Neighbors": knearest,
    "SVM": svm_model,
    "Naive Bayes": naive
}

## Evaluate models
# average accuracy
avg_accuracy = {}
avg_accuracy["Multilayer Perceptron"] = 0.920 # mlp.py

for name, m in models.items():
  scores = cross_val_score(m, X, y, cv=100, n_jobs=-1) # 100-fold cross-validation
  avg_accuracy[name] = sum(scores) / len(scores)

# display avg_accuracy
for i in sorted(avg_accuracy, key=avg_accuracy.get, reverse=True):
  print("%-30s%-20s" % (i, "{:.2f}".format(avg_accuracy[i]*100)))

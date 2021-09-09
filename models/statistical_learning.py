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
X = np.load('data/phase1/samples.npy')
y = np.load('data/phase1/labels.npy')

## Initialize models
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, max_features=55, max_leaf_nodes=28, min_impurity_decrease=0, min_samples_leaf=2)
logreg = LogisticRegression(C=0.001, penalty='none', solver='newton-cg')
ridge = RidgeClassifier(alpha=0.01, solver='sparse_cg')
svm_model = svm.SVC(degree=0,C=100,gamma=0.0001,kernel='linear')
knearest = neighbors.KNeighborsClassifier(leaf_size=1,n_neighbors=18,p=1)
naive = GaussianNB(var_smoothing=1.0)

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

for name, m in models.items():
  scores = cross_val_score(m, X, y, cv=10, n_jobs=-1) # 100-fold cross-validation
  avg_accuracy[name] = sum(scores) / len(scores)

# display avg_accuracy
for i in sorted(avg_accuracy, key=avg_accuracy.get, reverse=True):
  print("%-30s%-20s" % (i, "{:.2f}".format(avg_accuracy[i]*100)))

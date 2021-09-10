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
from sklearn.model_selection import train_test_split
import csv
import numpy as np

OUTPUT = "data/coefficients/coefficients.csv"
X = np.load('data/original dataset/samples.npy')
y = np.load('data/original dataset/labels.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

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

# fit models
for name, m in models.items():
  m.fit(X_train, y_train)

# coefficients dictionary
coeffs = {}
for name, m in models.items():
  try:
    coeffs[name] = m.coef_.tolist()[0] # NOTE: first two coeffs are for race, age
  except (AttributeError):
    continue

print(coeffs.keys())

# write coefficient arrays to csv
with open(OUTPUT, 'w', newline='') as csvfile:
  csvWriter = csv.writer(csvfile)
  
  # header
  header = ["model", "race", "age"]
  for i in range(68):
    header.append('x' + str(i))
    header.append('y' + str(i))
  csvWriter.writerow(header)

  # store coefficients
  for name, coeff_arr in coeffs.items():
    row = [name]
    for c in coeff_arr:
      row.append(c)
    csvWriter.writerow(row)

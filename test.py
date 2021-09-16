from src.create_dataset.load_data import load_data
from src.train.k_fold_cross_validation import k_fold_prediction
from sklearn.tree import DecisionTreeClassifier

SRC = "L:\\Autonomic Medicine\\Dysmorphology Photos\\Facial recognition project\\Angeli scripts\\experimental\\cchs-experimental\\data\\datasets\\stratified_subjects\\train"
K = 10

data = load_data(SRC, K)
print(data)

cnf = k_fold_prediction(DecisionTreeClassifier(), data)
print(cnf)
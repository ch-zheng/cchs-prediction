from ridge import Ridge
from logistic import Logistic
from svm import SVM
from neighbors import Neighbors
from bayes import NaiveBayes
from tree import DecisionTree
from utils import oversample
from pathlib import Path
import json
import numpy as np

# Load data
samples = np.load('data/original/samples.npy')
labels = np.load('data/original/labels.npy')
X, y = oversample(samples, labels)

# Model list
models = {
    'ridge': Ridge(),
    'logistic': Logistic(),
    'svm': SVM(),
    'neighbors': Neighbors(),
    'bayes': NaiveBayes(),
    'tree': DecisionTree()
}

for name, model in models.items():
    # Hyperparameter search
    parameters = model.grid_search(X, y)
    print(name.capitalize(), 'parameters:', parameters)
    # Evaluation
    model.load_hyperparams(parameters)
    score = model.cross_validate(X, y)
    print(name.capitalize(), 'score:', score)
    # Save hyperparameters
    with open(Path('hyperparameters', name + '.json'), 'w') as f:
        json.dump(parameters, f)

import utils
import zoo
from pathlib import Path
import json
import numpy as np

# Load data
X = np.load('data/individualized/samples.npy')
y = np.load('data/individualized/labels.npy')
X, y = utils.augment(X, y)
splits = utils.split(10, X, y)
X = np.delete(X, 0, 1)

# Model list
models = {
    'ridge': zoo.Ridge(),
    'logistic': zoo.Logistic(),
    'svm': zoo.SVM(),
    'neighbors': zoo.Neighbors(),
    'bayes': zoo.NaiveBayes(),
    'tree': zoo.DecisionTree()
}

for name, model in models.items():
    # Load hyperparameters
    with open(Path('hyperparameters', name + '.json')) as f:
        parameters = json.load(f)
    model.load_hyperparams(parameters)
    # Evaluation
    scores = model.cross_validate(X, y, splits)
    print(name, scores)

# Local
import utils
#import transform
import zoo
# Native
from pathlib import Path
import json
# External
import numpy as np

# Load data
# Row format: (subject, race, age, landmarks...)
'''
X = np.load('data/individualized/samples.npy')
y = np.load('data/individualized/labels.npy')
X, y = utils.augment(X, y, 3)
X = transform.relativize(X, 3)
X = transform.normalize(X, 3)
splits = utils.split(10, X, y)
X = np.delete(X, 0, 1)
'''

X = np.load('data/pruned/samples.npy')
y = np.load('data/pruned/labels.npy')
#X, y = utils.augment(X, y, 3)
splits = utils.split(10, X, y)
subjects = X[:, 0]
X = np.delete(X, 0, 1)

# Model list
models = {
    'ridge': zoo.Ridge(),
    #'logistic': zoo.Logistic(),
    'svm': zoo.SVM(),
    'neighbors': zoo.Neighbors(),
    #'bayes': zoo.NaiveBayes(),
    'tree': zoo.DecisionTree()
}

def grid_search():
    for name, model in models.items():
        print('Grid searching for', name)
        # Hyperparameter search
        parameters = model.grid_search(X, y, cv=splits)
        print(name, 'parameters:', parameters)
        # Save hyperparameters
        with open(Path('hyperparameters', name + '.json'), 'w') as f:
            json.dump(parameters, f)

def evaluate():
    for name, model in models.items():
        print('Evaluating', name)
        # Load hyperparameters
        with open(Path('hyperparameters', name + '.json')) as f:
            parameters = json.load(f)
        model.load_hyperparams(parameters)
        # Evaluation
        scores = model.cross_validate(X, y, splits)
        print(name, scores)

def save():
    for name, model in models.items():
        print('Saving', name)
        # Load hyperparameters
        with open(Path('hyperparameters', name + '.json')) as f:
            parameters = json.load(f)
        model.load_hyperparams(parameters)
        model.fit(X, y)
        model.save(Path('pretrained', name + '.pickle'))

def test():
    for name, model in models.items():
        print('Evaluating', name)
        # Load hyperparameters
        with open(Path('hyperparameters', name + '.json')) as f:
            parameters = json.load(f)
        model.load_hyperparams(parameters)
        # Evaluation
        scores = model.cross_validate(X, y, splits)
        print(name, scores)

#grid_search()
#evaluate()
#save()
test()

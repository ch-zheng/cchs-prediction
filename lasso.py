# Local
import utils
# Native
# External
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Load data
X = np.load('data/individualized/samples.npy')
y = np.load('data/individualized/labels.npy')
X, y = utils.augment(X, y)
#splits = utils.split(10, X, y)
X = np.delete(X, 0, 1)

model = Lasso(max_iter=10000)
params = {
    'alpha': (1, 0.5),
    'selection': ('cyclic', 'random')
}
searcher = GridSearchCV(model, params, scoring='balanced_accuracy', n_jobs=-1, verbose=4)
searcher.fit(X, y)

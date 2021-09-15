# Native
import pickle
import statistics
import os
from abc import ABC, abstractmethod
from typing import Tuple
# External
from sklearn.model_selection import cross_val_score, GridSearchCV

class Model(ABC):
    # ML workflow
    @abstractmethod
    def fit(self, X, y):
        pass
    @abstractmethod
    def predict(self, X):
        pass
    @abstractmethod
    def score(self, X, y) -> float:
        pass
    # Disk ops
    @abstractmethod
    def save(self, file: os.PathLike):
        pass
    @abstractmethod
    def load(self, file: os.PathLike):
        pass

# Scikit model
class SKModel(Model):
    def __init__(self):
        self.test_params = {}
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def score(self, X, y) -> float:
        return self.model.score(X, y)
    def cross_validate(self, X, y, cv=10) -> dict:
        scores = {
            'recall': (0, 0),
            'precision': (0, 0),
            'accuracy': (0, 0),
            'balanced_accuracy': (0, 0),
        }
        for k in scores.keys():
            s = cross_val_score(self.model, X, y, scoring=k, cv=cv, n_jobs=-1)
            scores[k] = statistics.mean(s), statistics.stdev(s)
        return scores
    def grid_search(self, X, y, cv=10, params=None) -> dict:
        if params is None:
            params = self.test_params
        searcher = GridSearchCV(self.model, params, scoring='balanced_accuracy', n_jobs=-1, refit=False, cv=cv)
        searcher.fit(X, y)
        return searcher.best_params_
    def load_hyperparams(self, params: dict):
        self.model.set_params(**params)
    def save(self, file: os.PathLike):
        with open(file, 'wb') as f:
            pickle.dump(self.model, f)
    def load(self, file: os.PathLike):
        with open(file, 'rb') as f:
            self.model = pickle.load(f)

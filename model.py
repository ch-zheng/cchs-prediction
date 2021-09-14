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
    @abstractmethod
    def cross_validate(self, X, y, k: int = 10) -> Tuple[float, float]:
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
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def score(self, X, y) -> float:
        return self.model.score(X, y)
    def cross_validate(self, X, y, k: int = 10) -> Tuple[float, float]:
        scores = cross_val_score(self.model, X, y, cv=k, n_jobs=-1)
        return statistics.mean(scores), statistics.stdev(scores)
    def grid_search(self, params: dict, X, y) -> dict:
        searcher = GridSearchCV(self.model, params, n_jobs=-1, refit=False, cv=10)
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

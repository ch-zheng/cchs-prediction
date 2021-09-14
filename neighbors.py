from model import SKModel
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Description: K-Nearest Neighbors

class Neighbors(SKModel):
    def __init__(self):
        self.model = KNeighborsClassifier(n_jobs=-1)
    def grid_search(self, X, y) -> dict:
        parameters = {
            'n_neighbors': list(range(1, 10)),
            'weights': ('uniform', 'distance'),
            'p': list(range(1, 4))
        }
        return super().grid_search(parameters, X, y)

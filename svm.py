from model import SKModel
import numpy as np
from sklearn.svm import LinearSVC

# Description: Support Vector Machine

class SVM(SKModel):
    def __init__(self):
        self.model = LinearSVC(dual=False, max_iter=10000)
    def grid_search(self, X, y) -> dict:
        parameters = {
            'C': np.logspace(0, 9, num=10, base=2)
        }
        return super().grid_search(parameters, X, y)

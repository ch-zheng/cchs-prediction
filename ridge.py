from model import SKModel
import numpy as np
from sklearn.linear_model import RidgeClassifier

# Description: Ridge regression classifier

class Ridge(SKModel):
    def __init__(self):
        self.model = RidgeClassifier(solver='saga')
    def grid_search(self, X, y) -> dict:
        parameters = {
            'alpha': np.logspace(-8, 8, num=17, base=2)
        }
        return super().grid_search(parameters, X, y)

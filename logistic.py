from model import SKModel
import numpy as np
from sklearn.linear_model import LogisticRegression

# Description: Logistic regression classifier

class Logistic(SKModel):
    def __init__(self):
        self.model = LogisticRegression(solver='saga', max_iter=10000, n_jobs=-1, multi_class='ovr')
    def grid_search(self, X, y) -> dict:
        parameters = {
            'C': np.logspace(-8, 8, num=17, base=2)
        }
        return super().grid_search(parameters, X, y)

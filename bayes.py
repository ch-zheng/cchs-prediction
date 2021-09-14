from model import SKModel
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Description: Gaussian Naive Bayes classifier

class NaiveBayes(SKModel):
    def __init__(self):
        self.model = GaussianNB()
    def grid_search(self, X, y) -> dict:
        parameters = {
            'var_smoothing': np.logspace(-16, 0, num=17)
        }
        return super().grid_search(parameters, X, y)

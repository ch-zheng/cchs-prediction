from model import SKModel
from sklearn.tree import DecisionTreeClassifier

# Description: Decision Tree
# Accuracy: 70.04%

class DecisionTree(SKModel):
    def __init__(self):
        self.model = DecisionTreeClassifier()
    def grid_search(self, X, y) -> dict:
        parameters = {
            'criterion': ('gini', 'entropy'),
            'max_depth': list(range(2, 8)),
            'min_samples_split': list(range(2, 8)),
            'min_samples_leaf': list(range(1, 8)),
            'max_features': ('sqrt', 'log2', None)
        }
        return super().grid_search(parameters, X, y)

# CCHS Prediction
Predict whether an individual has CCHS based on frontal facial landmarks.

## Performance
| Model | Accuracy |
| --- | --- |
| Multilayer Perceptron | 90.00% |
| Ridge regression | 81.94% |
| Logistic regression | 78.00% |
| K-nearest neighbors | 71.52% |
| Decision tree | 70.04% |
| Support Vector Machine | 68.68% |
| Naive Bayes | 53.77% |

Accuracy measured with 100-fold cross-validation.
Default model parameters used, no hyperparameter tuning done.

## File Contents
* cchs\_statistical\_learning.ipynb
	1. Decision Tree
	2. Logistic regression
	3. LASSO regression
	4. Linear regression
	5. Polynomial linear regression
### Utility
* tabulate.py: Landmark photos & write to CSV file
* encode.py: Serialize CSV entries as NumPy array files
* decode.py: Find CSV table entry based on the initial entries of an encoded row.
### Models
* ridge.py: Ridge regression classifier
* log-reg.py: Logistic regression classifier
* svm.py: Support Vector Machine
* tree.py: Decision tree
* k-neighbors.py: K-nearest neighbors
* gaussian.py: Gaussian process (_nonfunctional_)
* naive-bayes.py: Naive Bayes classifier
* mlp.py: Multilayer Perceptron

## Data Content
Contains\
    - CSV file with samples and labels\
    - Master list of samples and labels in samples.npy and labels.npy\
    - Test and training samples and labels in respective folders test/ and training/ labeled similarly

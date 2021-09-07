# CCHS Prediction
Predict whether an individual has CCHS based on frontal facial landmarks.

## Performance
### First trial
| Model | Accuracy |
| --- | --- |
| Multilayer Perceptron | 90.00% |
| Ridge regression | 81.94% |
| Logistic regression | 78.00% |
| K-nearest neighbors | 71.52% |
| Decision tree | 70.04% |
| Support Vector Machine | 68.68% |
| Naive Bayes | 53.77% |

## Second trial
| Model | Accuracy |
| --- | --- |
| Multilayer Perceptron | 92.00% |
| Ridge regression | ---% |
| Logistic regression | ---% |
| K-nearest neighbors | ---% |
| Decision tree | ---% |
| Support Vector Machine | ---% |
| Naive Bayes | ---% |

Accuracy measured with 100-fold cross-validation.
Default model parameters used, no hyperparameter tuning done.

## File Contents
* statistical\_learning.ipynb
	1. Decision tree
	2. Logistic regression
	3. LASSO regression
	4. Polynomial regression
    5. Ridge regression
    6. K-nearest neighbors
    7. Support vector machine
    8. Naive bayes
### Utility
* tabulate.py: Landmark photos & write to CSV file.
* encode.py: Serialize CSV entries as NumPy array files.
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
### coefficients/
* graph_coefficients.py: Display GUI weighting coefficients from regression models.
### filter photos/, proportions/
* filter_funny_faces.ipynb: rotate images upright, determine photos with dlib-undetectable faces
* encode_modified.py: modified encode.py to work on cchs photos only
* tabulate_modified.py: modified tabulate.py to work on cchs photos only
* landmark_proportions.py: perform proportion manipulations from points in samples_final.csv

## Data Content
### samples.npy, labels.npy, samples.csv
* master samples, labels as well as test/ and training/ samples, labels read from samples.csv
### coefficients/
* coefficients.csv: coefficients corresponding to each regression model
* gui points.csv: landmarks on face.png to use for GUI
### filter photos/, proportions/
* samples_final.csv: (to delete)
* proportions_manipulations.xlsx: manipulate x, y points to determine "funny faces"
* landmarked.txt: list of photos with faces detectable by dlib
* unrecognized.txt: list of photos with faces undetectable by dlib

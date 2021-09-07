# CCHS Prediction
Predict whether an individual has CCHS based on frontal facial landmarks.

## Performance
<table>
    <tr>
        <th rowspan = 2>Model</th>
        <th colspan=2>Accuracy</th>
    </tr>
    <tr>
        <th>First Trial</th>
        <th>Second Trial</th>
    </tr>
    <tr>
        <td>Multilayer Perceptron</td>
        <td>90.00%</td>
        <td>92.00%</td>
    </tr>
    <tr>
        <td>Ridge regression</td>
        <td>81.94%</td>
        <td>82.73%</td>
    </tr>
    <tr>
        <td>Logistic regression</td>
        <td>78.05%</td>
        <td>79.29%</td>
    </tr>
    <tr>
        <td>K-nearest neighbors</td>
        <td>71.52%</td>
        <td>70.38%</td>
    </tr>
    <tr>
        <td>Decision tree</td>
        <td>70.45%</td>
        <td>71.84%</td>
    </tr>
    <tr>
        <td>Support Vector Machine</td>
        <td>68.68%</td>
        <td>73.38%</td>
    </tr>
    <tr>
        <td>Naive Bayes</td>
        <td>53.77%</td>
        <td>53.64%</td>
    </tr>
</table>
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

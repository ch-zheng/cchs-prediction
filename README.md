# CCHS Prediction
Predict whether an individual has CCHS based on frontal facial landmarks.

## Performance
<table>
    <tr>
        <th rowspan=2>Model</th>
        <th colspan=4>Accuracy</th>
    </tr>
    <tr>
        <th>Initial Results</th>
        <th>Pruning Dataset</th>
        <th>With Optimization</th>
        <th>Phase 1 Landmark Reduction</th>
    </tr>
    <tr>
        <td>Multilayer Perceptron</td>
        <td>90.00%</td>
        <td>92.00%</td>
        <td>---%</td>
        <td>---%</td>
    </tr>
    <tr>
        <td>Ridge regression</td>
        <td>81.94%</td>
        <td>82.73%</td>
        <td>88.16%</td>
        <td>---%</td>
    </tr>
    <tr>
        <td>Logistic regression</td>
        <td>78.05%</td>
        <td>79.29%</td>
        <td>88.33%</td>
        <td>---%</td>
    </tr>
    <tr>
        <td>K-nearest neighbors</td>
        <td>71.52%</td>
        <td>70.38%</td>
        <td>75.96%</td>
        <td>---%</td>
    </tr>
    <tr>
        <td>Decision tree</td>
        <td>70.45%</td>
        <td>71.84%</td>
        <td>77.43%</td>
        <td>---%</td>
    </tr>
    <tr>
        <td>Support Vector Machine</td>
        <td>68.68%</td>
        <td>73.38%</td>
        <td>88.00%</td>
        <td>---%</td>
    </tr>
    <tr>
        <td>Naive Bayes</td>
        <td>53.77%</td>
        <td>53.64%</td>
        <td>73.38%</td>
        <td>---%</td>
    </tr>
</table>
Accuracy measured with 100-fold cross-validation.
Default model parameters used, no hyperparameter tuning done.
Phase 1 removed landmarks 1, 3, 5, 11, 13, 15, 18, 25, 28, reducing our landmark total from 68 to 59 points.

## File Contents
### Utility
* tabulate.py: Landmark photos & write to CSV file.
* encode.py: Serialize CSV entries as NumPy array files.
* decode.py: Find CSV table entry based on the initial entries of an encoded row.
### Models
* Ridge regression classifier
* Logistic regression classifier
* Support Vector Machine
* Decision tree
* K-nearest neighbors
* Gaussian process (_nonfunctional_)
* Naive Bayes classifier
* mlp.py: Multilayer Perceptron
### Evaluation
* statistical\_learning.py: create aforemention models (with exception to MLP) and evaluate accuracy with 100-fold cross-validation.

### coefficients/
* generate_coefficients.py: Generate coefficients from regression models to visually display.
* graph_coefficients.py: Display GUI weighting coefficients from regression models.
### filter photos/
* filter_funny_faces.ipynb: rotate images upright, determine photos with dlib-undetectable faces.
* landmark_proportions.py: perform proportion manipulations from points in samples_final.csv.

## Data Content
### samples.npy, labels.npy, samples.csv
* master samples, labels as well as test/ and training/ samples, labels read from samples.csv.
### coefficients/
* coefficients.csv: coefficients corresponding to each regression model.
* gui points.csv: landmarks on face.png to use for GUI.
* face.png: sample photo to use for GUI.
### filter photos/
* proportions.csv: generated csv file of desired proportions described in filter photos/landmark_proportions.py
* proportions_manipulations.xlsx: manipulate x, y points to determine "funny faces"
* landmarked.txt: list of photos with faces detectable by dlib
* unrecognized.txt: list of photos with faces undetectable by dlib

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
        <td>87.20%</td>
    </tr>
    <tr>
        <td>Logistic regression</td>
        <td>78.05%</td>
        <td>79.29%</td>
        <td>88.33%</td>
        <td>87.82%</td>
    </tr>
    <tr>
        <td>K-nearest neighbors</td>
        <td>71.52%</td>
        <td>70.38%</td>
        <td>75.96%</td>
        <td>75.95%</td>
    </tr>
    <tr>
        <td>Decision tree</td>
        <td>70.45%</td>
        <td>71.84%</td>
        <td>77.43%</td>
        <td>76.12%</td>
    </tr>
    <tr>
        <td>Support Vector Machine</td>
        <td>68.68%</td>
        <td>73.38%</td>
        <td>88.00%</td>
        <td>87.12%</td>
    </tr>
    <tr>
        <td>Naive Bayes</td>
        <td>53.77%</td>
        <td>53.64%</td>
        <td>73.38%</td>
        <td>73.38%</td>
    </tr>
</table>
Accuracy measured with 100-fold cross-validation.<br>
Default model parameters used, no hyperparameter tuning done.<br>
Phase 1 removed landmarks 1, 3, 5, 11, 13, 15, 18, 25, 28, reducing our landmark total from 68 to 59 points.

## Code Contents
### utility/
* tabulate.py: Landmark photos & write to CSV file.
* encode.py: Serialize CSV entries as NumPy array files.
* decode.py: Find CSV table entry based on the initial entries of an encoded row.
### models/
* statistical_learning.py: Create optimized models, evaluate accuracy with 100-fold cross-validation.
  * Ridge regression classifier
  * Logistic regression classifier
  * Support vector machine
  * Decision tree
  * K-nearest neighbors
  * Naive Bayes classifier
* mlp.py: Implement Multilayer Perceptron model.
* mlp_cross_validation.py: Implement MLP with 10-fold cross-validation and 60,000 epochs for each evaluation.

### coefficients/
* generate_coefficients.py: Generate coefficients from regression models to visually display.
* graph_coefficients.py: Display GUI weighting coefficients from regression models.

### filter photos/
* filter_funny_faces.ipynb: Rotate images upright, determine photos with dlib-undetectable faces.
* landmark_proportions.py: Perform proportion manipulations from points in samples_final.csv.

## Data Contents
* samples.csv: Contains all CCHS and control subject race, age group, and landmark labels.

### coefficients/
* coefficients.csv: Coefficients corresponding to each regression model.
* gui points.csv: Landmarks on face.png to use for GUI.
* face.png: Sample photo to use for GUI.

### filter photos/
* proportions.csv: Generated csv file of desired proportions described in filter photos/landmark_proportions.py
* proportions_manipulations.xlsx: Manipulate x, y points to determine "funny faces"
* landmarked.txt: List of photos with faces detectable by dlib
* unrecognized.txt: List of photos with faces undetectable by dlib

### grid-search/
* best_hyperparameters.txt: List of best hyperparameters for models in statistical_learning.py based on grid-search results on original dataset.
* grid_search_results.csv: List results of hyperparameter grid search on statistical learning models as more hyperparameters are evaluated with 100-fold cross-validation.

### original dataset/
* samples.npy: Encoded samples from samples.csv sans modification.
* labels.npy: Encoded labels from samples.csv sans modification.

### phase1/
* samples_phase1.npy: Encoded samples from samples.csv, omitting undesired landmarks indicated for phase 1.
* labels_phase1.npy: Encoded labels from samples.csv, omitting undesired landmarks indicated for phase 1.

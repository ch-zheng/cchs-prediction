# CCHS Prediction
Predict whether an individual has CCHS based on frontal facial landmarks.

## File Content
### decode.py
Find CSV table entry based on the initial entries of an encoded row

### encode.py
Encode CSV table into NumPy array files

### tabulate.py
Landmarks frontal photos & writes to a CSV file

### mlp.py
Train a multilayer perceptron neural net

### cchs_statistical_learning.ipynb
Contains the following statistical learning algorithms:\
    1. Decision Tree\
    2. Logistic Regression\
    3. LASSO Regression

## Data Content
Contains\
    - CSV file with samples and labels\
    - Master list of samples and labels in samples.npy and labels.npy\
    - Test and training samples and labels in respective folders test/ and training/ labeled similarly

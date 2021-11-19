# CCHS Prediction
Predict whether an individual has CCHS based on frontal facial landmarks.

## File contents
* encode.py: Convert .CSV table to .NPY file
* main.py: tune, train, & evaluate models
* missed.py: Print expected & actual predictions
* mlp.py: Multilayer perceptron
* mlp-cv.py: MLP cross-validation
* model.py: Library of models
* prune.py: Remove landmark points from data
* plot.py: Script to visualize landmark locations
* tabulate.py: Landmark images & write to .CSV table
* transform.py: Relativized distance metric
* utils.py: Data transformation functions

### data
* pruned/: Dataset with landmarks removed
* individualized/: Dataset with photos assigned to subjects

### hyperparameters
Contains saved model hyperparameters.

### pretrained
Contains saved model binaries.

### results
Contains saved model summaries.

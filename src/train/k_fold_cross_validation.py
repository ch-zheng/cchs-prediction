from sklearn.metrics import confusion_matrix
import sys
# Internal
from src.models.initialize_models import initialize_model

def k_fold_prediction(data):
    cnf_matrices = {} # model_name:sum(cnf_matrix) of k trials
    for df in data:
        models = initialize_model()
        for model_name, m in models.items():
            try:
                m.fit(df.X_train, df.y_train.values.ravel())
                y_pred = m.predict(df.X_test)
                cnf_matrix = confusion_matrix(df.y_test, y_pred, labels=[0,1])

                if model_name not in cnf_matrices.keys():
                    cnf_matrices[model_name] = cnf_matrix
                else:
                    cnf_matrices[model_name] += cnf_matrix
            except (ValueError):
                print("ERROR:", model_name)
    return cnf_matrices

def k_fold_prediction(m, data):
    cnf_mtx = 0 # model_name:sum(cnf_matrix) of k trials
    for df in data:
        m.fit(df.X_train, df.y_train.values.ravel())
        y_pred = m.predict(df.X_test)
        cnf_matrix = confusion_matrix(df.y_test, y_pred, labels=[0,1])

        cnf_mtx += cnf_matrix
    return cnf_mtx
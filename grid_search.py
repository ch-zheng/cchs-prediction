from src.models.hyperparameter_tuning import *
from src.create_dataset.load_data import load_stratify, load_phase1, load_base
import os
print(os.getcwd())
K = 10
DATASETS = ['stratified_subjects', 'phase1', 'base']
DATASET = DATASETS[1]
SRC = os.path.join("L:\\Autonomic Medicine\\Dysmorphology Photos\\Facial recognition project\\Angeli scripts\\experimental\\cchs-experimental\\data\\datasets", DATASET)
OUTPUT_FILE = "L:\\Autonomic Medicine\\Dysmorphology Photos\\Facial recognition project\\Angeli scripts\\experimental\\cchs-experimental\\data\\results\\grid_search\\grid_search_results1.csv"

def main():
    if DATASET == 'stratified_subjects':
        data = load_stratify(os.path.join(SRC, "train"), K)
    elif DATASET == 'phase1':
        data = load_phase1(SRC, K)
    elif DATASET == 'base':
        data = load_base(SRC, K)
    else:
        print("INVALID DATASET")
        exit(1)
    grid_searches = [grid_search_tree, grid_search_logreg, grid_search_ridge, grid_search_svm, grid_search_neighbors, grid_search_naive]

    for gs in grid_searches:
        print("~~~~~~~~~~~~~STARTING ", gs)
        gs(OUTPUT_FILE, data)
        print("~~~~~~~~~~~~~DONE", gs)

if __name__ == "__main__":
    main()
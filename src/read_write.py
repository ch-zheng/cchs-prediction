import pickle
import pandas as pd
import csv

def to_pickle(dst, data):
    with open(dst, "wb") as handle:
        pickle.dump(data, handle,)

def load_pickle(src):
    data = pd.read_pickle(src)
    return data

def write_csv_header(dst):
    with open(dst, 'a', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)

        header = ["model", "race", "age"]
        for i in range(68):
            header.append('x' + str(i))
            header.append('y' + str(i))
        csvWriter.writerow(header)

def to_csv(dst, row):
    with open(dst, 'a', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow(row)
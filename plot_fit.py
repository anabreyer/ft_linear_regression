import json
import matplotlib.pyplot as plt
import csv

def load_data(path="data/data.csv"):
    X, Y = [], []
    with open(path, "r") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            X.append(float(row[0]))
            Y.append(float(row[1]))
        return X,y


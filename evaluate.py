# evaluate.py
import json
import csv
import math
from utils import *

def load_data(path="data/data.csv"):
    X, Y = [], []
    with open(path, "r") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            X.append(float(row[0]))
            Y.append(float(row[1]))
    return X, Y

def load_thetas(path="model/thetas.json"):
    with open(path, "r") as f:
        d = json.load(f)
    return float(d["theta0"]), float(d["theta1"])

def mean(values):
    return sum(values) / len(values)

def metrics(X, Y, t0, t1):
    m = len(X)
    preds = [estimate_price(x, t0, t1) for x in X]
    errors = [p - y for p, y in zip(preds, Y)]

    mae = sum(abs(e) for e in errors) / m
    rmse = math.sqrt(sum(e*e for e in errors) / m)

    y_bar = mean(Y)
    ss_tot = sum((y - y_bar) ** 2 for y in Y) #how much the real prices vary (compared to their mean).
    ss_res = sum((y - p) ** 2 for y, p in zip(Y, preds)) #how much error remains after using the model.
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return mae, rmse, r2

if __name__ == "__main__":
    X, Y = load_data()
    t0, t1 = load_thetas()
    mae, rmse, r2 = metrics(X, Y, t0, t1)

    print(f"Thetas: theta0={t0:.4f}, theta1={t1:.8f}")
    print(f"MAE : {mae:.2f}") #Mean absolute error - average price difference ex: 500 = 500 euros off
    print(f"RMSE: {rmse:.2f}") #Root mean squared error - calculate big mistakes tend to be
    print(f"R^2 : {r2:.4f} (1=perfect, 0=mean-only model)") #what % of variation in prices the model explains

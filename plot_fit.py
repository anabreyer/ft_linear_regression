import json
import matplotlib.pyplot as plt
import csv
from utils import *

def load_data(path="data/data.csv"):
    X, Y = [], []
    with open(path, "r") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            X.append(float(row[0]))
            Y.append(float(row[1]))
        return X,Y

def load_thetas(path="model/thetas.json"):
    with open(path, "r") as f:
        d = json.load(f)
    return float(d["theta0"]), float(d["theta1"])

if __name__ == "__main__":
    # data + model
    X, Y = load_data()
    t0, t1 = load_thetas()

    # scatter of points
    plt.figure()
    plt.scatter(X, Y, label="Data", s=20)

    # regression line across the range of X
    x_min, x_max = min(X), max(X)
    xs = [x_min, x_max]
    ys = [estimate_price(x_min, t0, t1), estimate_price(x_max, t0, t1)]
    plt.plot(xs, ys, label=f"Fit: y = {t0:.1f} + {t1:.5f}·x")

    plt.title("Mileage vs. Price with Linear Regression")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

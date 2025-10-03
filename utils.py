import csv
from train import *

def load_data(filename):
    mileages = []
    prices = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header line (mileage,price)
        for row in reader:
            mileage = float(row[0])
            price = float(row[1])
            mileages.append(mileage)
            prices.append(price)
    return mileages, prices

def compute_cost(X, Y, theta0, theta1):
    # Mean Squared Error (MSE) / 2 (common in GD derivation)
    m = len(X)
    s = 0.0
    for i in range(m):
        pred = estimate_price(X[i], theta0, theta1)
        err = pred - Y[i]
        s += err * err
    return s / (2 * m)

def save_thetas(theta0, theta1, path="model/thetas.json"):
    with open(path, "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)

def zscore_fit(values):
    #calculate de mean and standart deviation
    m = len(values)
    mean = sum(values) / m
    var = sum((v - mean) ** 2 for v in values) / m  # population variance
    std = var ** 0.5
    if std == 0:
        std = 1.0
    return mean, std

def zscore_transform(values, mean, std):
    return [(v - mean) / std for v in values]

def backconvert_thetas(a_norm, b_norm, x_mean, x_std, y_mean, y_std):
    # normalized model: y' = a + b * x'
    # original units:   y  = theta0 + theta1 * x
    theta1 = (y_std * b_norm) / x_std
    theta0 = y_mean + y_std * a_norm - theta1 * x_mean
    return theta0, theta1

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

import json
import math
from utils import *

def train(X, Y, learning_rate=0.05, iterations=6000, verbose_every=300):
    m = len(X)

    #normalize X and Y
    x_mean, x_std = zscore_fit(X)
    y_mean, y_std = zscore_fit(Y)
    Xn = zscore_transform(X, x_mean, x_std)
    Yn = zscore_transform(Y, y_mean, y_std)

    a = 0.0
    b = 0.0

    for it in range(1, iterations + 1):
        error0 = 0.0
        error1 = 0.0
        # accumulate gradients
        for i in range(m):
            pred = a + b * Xn[i]
            diff = pred - Yn[i]
            error0 += diff
            error1 += diff * Xn[i]

        # simultaneous update
        a -= (learning_rate * (error0 / m))
        b -= (learning_rate * (error1 / m))

        if verbose_every and it % verbose_every == 0:
            # cost in normalized space for monitoring
            # reuse your compute_cost by passing a/b as thetas on Xn/Yn
            norm_cost = compute_cost(Xn, Yn, a, b)
            # show back_converted thetas so numers makes sense
            t0_cur, t1_cur = backconvert_thetas(a, b, x_mean, x_std, y_mean, y_std)
            print(f"[iter {it}] norm_cost={norm_cost:.6f}  theta0={t0_cur:.4f}  theta1={t1_cur:.8f}")

    theta0, theta1 = backconvert_thetas(a, b, x_mean, x_std, y_mean, y_std)
    return theta0, theta1

if __name__ == "__main__":
    X, Y = load_data("data/data.csv")
    print("Loaded data points:", len(X))
    print("First 5 examples")
    for i in range(min(5, len(X))):
        print(f"Mileage: {X[i]}, Price: {Y[i]}")

    # Train
    theta0, theta1 = train(X, Y, learning_rate=0.05, iterations=3000, verbose_every=300)
    print(f"\nTrained thetas: theta0={theta0:.4f}, theta1={theta1:.8f}")

    # Save for predict.py
    save_thetas(theta0, theta1)
    print("Saved to model/thetas.json")

    # Quick sanity prediction on a sample mileage
    sample = 50000.0
    print(f"Example: mileage={sample:.0f} -> price≈ {estimate_price(sample, theta0, theta1):.2f}")

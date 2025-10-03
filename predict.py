import json
from utils import *

def load_thetas(path="model/thetas.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        #ensures numeric type
        theta0 = float(data.get("theta0", 0.0))
        theta1 = float(data.get("theta1", 0.0))
        return theta0, theta1, True #True - loaded from file
    except FileNotFoundError:
        #Before training, both are ) as per project
        return 0.0, 0.0, False
    except Exception:
        #Any parse error -> safe fall back
        return 0.0, 0.0, False

if __name__ == "__main__":
    theta0, theta1, loaded = load_thetas()

    if not loaded:
        print("Model file not found or unreadable. Using default theta0=0, theta1=0.\n"
              "Tip: run train.py to learn and save better parameters.")

    else:
        print(f"Loaded thetas: theta0={theta0:.4f}, theta1={theta1:.8f}")

    user_in = input("Enter mileage (km): ").strip()
    try:
        mileage = float(user_in)

    except ValueError:
        print("Please enter a valid number for mileage.")
        raise SystemExit(1)

    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price for mileage {mileage:.0f}: {price:.2f}")     

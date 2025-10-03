import json
from train.py import *

def load_thetas(path="model/theta.jason"):
    try:
        with open(path, "r") as f:
            
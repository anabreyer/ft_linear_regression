# ft_linear_regression

An introductory ML project: predict a car’s **price** from its **mileage** using **simple linear regression** trained with **gradient descent**.

## What you’ll build
- `train.py` — reads `data/data.csv`, trains θ₀, θ₁ via gradient descent (with z-score normalization for stability), saves to `model/thetas.json`.
- `predict.py` — interactive script: loads learned thetas and predicts price for a user-entered mileage.
- `plot_fit.py` — bonus: scatter of data + fitted regression line.
- `evaluate.py` — bonus: prints MAE, RMSE, and R².

## Project structure

ft_linear_regression/
├─ data/
│ └─ data.csv
├─ model/
│ └─ thetas.json # created by train.py
├─ train.py
├─ predict.py
├─ plot_fit.py # bonus
├─ evaluate.py # bonus
├─ utils.py # z-score helpers + backconversion
├─ requirements.txt
└─ Makefile


## Install

Use your system Python, or create a venv.

```bash
# system-wide (simple)
pip3 install -r requirements.txt

# or with a venv
make venv
source .venv/bin/activate
pip install -r requirements.txt

requirements.txt minimally includes:

matplotlib

(Training and prediction only use stdlib; matplotlib is for plot_fit.py.)
Usage

Train (reads data/data.csv, writes model/thetas.json):

make train

Predict (interactive; uses saved thetas):

make predict

Plot data + regression line (bonus):

make plot

Evaluate precision (bonus):

make eval

Clean caches:

make clean

Delete model (force retrain next time):

make reset

How it works

Model (hypothesis)
price^=θ0+θ1⋅mileage
price
​=θ0​+θ1​⋅mileage

Training

    We z-score normalize X (mileage) and Y (price):
    x′=x−μxσxx′=σx​x−μx​​, y′=y−μyσyy′=σy​y−μy​​

    Run gradient descent on normalized data for stability:
    y′=a+b⋅x′y′=a+b⋅x′

    Convert back to original units:
    θ1=σybσxθ1​=σx​σy​b​,
    θ0=μy+σya−θ1μxθ0​=μy​+σy​a−θ1​μx​

Loss (for monitoring)
J(θ0,θ1)=12m∑(y^−y)2
J(θ0​,θ1​)=2m1​∑(y
​−y)2
Interpreting results

    θ₀ (intercept): price at 0 km (brand-new approximation).

    θ₁ (slope): price change per km (usually negative).

    Reasonable predictions: higher mileage → lower price.

Troubleshooting

    NaN during training: You’re probably training on raw units with too large a learning rate. Use the provided normalization (default in train.py) or drastically lower learning_rate.

    model/thetas.json missing: Run make train once before predicting.

    Plot not showing: Ensure matplotlib is installed and you’re not in a headless environment, or use a backend like Agg to save figures instead of showing.

License

Educational use.
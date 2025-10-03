# ft_linear_regression

An introductory ML project: predict a car’s **price** from its **mileage** using **simple linear regression** trained with **gradient descent**.

## What you’ll build
- `train.py` — reads `data/data.csv`, trains θ₀, θ₁ via gradient descent (with z-score normalization for stability), saves to `model/thetas.json`.
- `predict.py` — interactive script: loads learned thetas and predicts price for a user-entered mileage.
- `plot_fit.py` — bonus: scatter of data + fitted regression line.
- `evaluate.py` — bonus: prints MAE, RMSE, and R².

## Project structure
```
ft_linear_regression/
├─ data/
│  └─ data.csv
├─ model/
│  └─ thetas.json                # created by train.py
├─ train.py
├─ predict.py
├─ plot_fit.py                   # bonus
├─ evaluate.py                   # bonus
├─ utils.py                      # z-score helpers + backconversion
├─ requirements.txt
└─ Makefile
```

## Install

Use your system Python, or create a venv.

```bash
# system-wide (simple)
pip3 install -r requirements.txt

# or with a venv
make venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` minimally includes:

```
matplotlib
```

(Training and prediction only use stdlib; `matplotlib` is for `plot_fit.py`.)

## Usage

Train (reads `data/data.csv`, writes `model/thetas.json`):
```bash
make train
```

Predict (interactive; uses saved thetas):
```bash
make predict
```

Plot data + regression line (bonus):
```bash
make plot
```

Evaluate precision (bonus):
```bash
make eval
```

Clean caches:
```bash
make clean
```

Delete model (force retrain next time):
```bash
make reset
```

## How it works

**Model (hypothesis)** \
$\
\widehat{price} = \theta_0 + \theta_1 \cdot mileage
$

**Training**  
- We z-score normalize X (mileage) and Y (price):  
  $\(x'=\frac{x-\mu_x}{\sigma_x}\), \(y'=\frac{y-\mu_y}{\sigma_y}\)$
- Run gradient descent on normalized data for stability:  
  $\(y' = a + b \cdot x'\)$
- Convert back to original units:  
  $\(\theta_1 = \frac{\sigma_y b}{\sigma_x}\)$,  
  $\(\theta_0 = \mu_y + \sigma_y a - \theta_1 \mu_x\)$

**Loss (for monitoring)**\
$\
J(\theta_0,\theta_1) = \frac{1}{2m}\sum (\widehat{y}-y)^2
$

## Interpreting results
- **θ₀ (intercept)**: price at 0 km (brand-new approximation).
- **θ₁ (slope)**: price change per km (usually negative).
- Reasonable predictions: higher mileage → lower price.

## Troubleshooting

- **NaN during training**: You’re probably training on raw units with too large a learning rate. Use the provided normalization (default in `train.py`) or drastically lower `learning_rate`.
- **`model/thetas.json` missing**: Run `make train` once before predicting.
- **Plot not showing**: Ensure `matplotlib` is installed and you’re not in a headless environment, or use a backend like `Agg` to save figures instead of showing.

## License
Educational use.

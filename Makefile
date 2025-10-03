# ===== ft_linear_regression — Makefile =====

PYTHON ?= python3
PIP    ?= pip3

# If you prefer a virtualenv, run: make venv && .venv/bin/python train.py
VENV_DIR := .venv
REQ      := requirements.txt

DATA     := data/data.csv
MODEL    := model/thetas.json

.PHONY: help setup venv install train predict plot eval clean reset

help:
	@echo "Targets:"
	@echo "  make setup     - Install Python deps from requirements.txt"
	@echo "  make venv      - Create a local virtualenv in .venv"
	@echo "  make train     - Train the model (reads $(DATA), writes $(MODEL))"
	@echo "  make predict   - Run interactive predictor (uses $(MODEL) if present)"
	@echo "  make plot      - Show data scatter + fitted regression line"
	@echo "  make eval      - Print MAE / RMSE / R^2 for current model"
	@echo "  make clean     - Remove __pycache__ and pyc files"
	@echo "  make reset     - Remove model file (forces retrain next time)"

setup:
	$(PIP) install -r $(REQ)

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activate with: source $(VENV_DIR)/bin/activate"
	@echo "Then run: pip install -r $(REQ)"

train: $(DATA)
	$(PYTHON) train.py

# NOTE: do NOT require $(MODEL) so predict can run with default thetas (0,0)
predict:
	$(PYTHON) predict.py

# plot and eval still require a trained model
plot: $(MODEL) $(DATA)
	$(PYTHON) plot_fit.py

eval: $(MODEL) $(DATA)
	$(PYTHON) evaluate.py

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} \; 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true

reset:
	@rm -f $(MODEL)
	@echo "Deleted $(MODEL). Run 'make train' to regenerate."

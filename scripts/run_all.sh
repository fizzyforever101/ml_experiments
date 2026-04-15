#!/bin/bash

echo "Running MIMIC baseline..."
python src/experiments/run_mimic_baseline.py

echo "Running fairness experiment..."
python src/experiments/run_mimic_fairness.py

echo "Running OLIVES..."
python src/experiments/run_olives_baseline.py

echo "Running OLIVES fairness..."
python src/experiments/run_olives_fairness.py

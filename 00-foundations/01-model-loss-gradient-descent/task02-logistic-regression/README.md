# Task 02: Logistic Regression (By Hand)

## Overview
Implementation of logistic regression from scratch using gradient descent — no autograd, no sklearn. Extends the linear regression foundation to binary classification via the sigmoid and cross-entropy loss.

## Task Description
See [task.md](task.md) for the task specification (S1-T2). *Note: Full S1-T2 details to be added from Week 1 Tasks document.*

## Directory Structure
```
task02-logistic-regression/
├── task.md              # Task specification (S1-T2)
├── README.md            # This file
├── implementation/      # Source code (logistic_regression.py)
├── tests/               # Unit tests
├── notes/               # Derivation notes, scratch work
└── experiments/         # Experiment logs, plots, outputs
```

## Expected Deliverables
- `implementation/logistic_regression.py` — Hand-coded gradient descent
- `experiments/` — Loss curves, decision boundary plots, printed logs
- `README.md` (this directory) — Derivation explanation in own words
- `notes/` — Hand-written gradient derivations (scanned or typed)

## Key Concepts to Demonstrate (Expected)
1. Sigmoid function and probability interpretation
2. Binary cross-entropy (log) loss formulation
3. Analytical gradients: ∂L/∂w, ∂L/∂b via chain rule
4. Gradient descent update rule for classification
5. Decision boundary visualization
6. Convergence verification on separable toy dataset
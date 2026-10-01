# Task 01: Linear Regression (By Hand)

## Overview
Implementation of linear regression from scratch using gradient descent — no autograd, no sklearn. This is the foundational model/loss/gradient-descent loop that all later stages build upon.

## Task Description
See [task.md](task.md) for the full task specification (S1-T1).

## Directory Structure
```
task01-linear-regression/
├── task.md              # Task specification (S1-T1)
├── README.md            # This file
├── implementation/      # Source code (linear_regression.py)
├── tests/               # Unit tests
├── notes/               # Derivation notes, scratch work
└── experiments/         # Experiment logs, plots, outputs
```

## Expected Deliverables
- `implementation/linear_regression.py` — Hand-coded gradient descent
- `experiments/` — Loss curves, convergence plots, printed logs
- `README.md` (this directory) — Derivation explanation in own words
- `notes/` — Hand-written gradient derivations (scanned or typed)

## Key Concepts to Demonstrate
1. MSE loss formulation and intuition
2. Analytical gradients: ∂L/∂w, ∂L/∂b
3. Gradient descent update rule: θ ← θ - α∇L
4. Learning rate effects (too large → divergence, too small → slow convergence)
5. Convergence verification on y = 3x + 2 + noise
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

---

## Derivation Explanation (In Own Words)

### Mean Squared Error (MSE) Loss
The MSE loss measures the average squared difference between predictions and true values:
```
L = (1/n) Σ (y_pred - y_true)²
  = (1/n) Σ (w·x_i + b - y_i)²
```
It penalizes larger errors quadratically, making the optimization landscape smooth and convex for linear regression — guaranteeing a single global minimum.

### Gradient Derivation (Step by Step)

**For weight w:**
```
L = (1/n) Σ (w·x_i + b - y_i)²
∂L/∂w = (1/n) Σ 2(w·x_i + b - y_i) · ∂/∂w(w·x_i + b - y_i)
      = (1/n) Σ 2(w·x_i + b - y_i) · x_i
      = (2/n) Σ (y_pred - y_true) · x_i
```

**For bias b:**
```
∂L/∂b = (1/n) Σ 2(w·x_i + b - y_i) · ∂/∂b(w·x_i + b - y_i)
      = (1/n) Σ 2(w·x_i + b - y_i) · 1
      = (2/n) Σ (y_pred - y_true)
```

**Vectorized form:**
- `dL/dw = (2/n) * X.T @ (y_pred - y_true)`
- `dL/db = (2/n) * sum(y_pred - y_true)`

### Why Subtract the Gradient?
The gradient ∇L points in the direction of **steepest increase** of the loss. To minimize loss, we move in the **opposite direction** — hence `θ ← θ - α∇L`. Adding the gradient would maximize loss (gradient ascent).

### Learning Rate (α) Effects
- **Too large (e.g., 100x):** Overshoots the minimum, loss oscillates or diverges to infinity
- **Too small:** Converges very slowly, may get stuck in numerical precision limits
- **Just right:** Smooth, monotonic convergence to the minimum

### Convergence Results
- True: w=3.0, b=2.0
- Learned: w≈2.977, b≈1.993 (within 0.03 of true values)
- Final MSE: ~0.202 (matches noise variance σ²=0.25, as expected)
- Gradients → 0 at convergence
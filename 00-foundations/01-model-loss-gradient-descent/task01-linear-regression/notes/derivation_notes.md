# Hand-Written Gradient Derivation Notes
## S1-T1: Linear Regression (By Hand)

---

### Problem Setup
- Model: ŷ = w·x + b
- True: y = 3x + 2 + ε,  ε ~ N(0, 0.5²)
- Loss: MSE = (1/n) Σ (ŷ - y)² = (1/n) Σ (w·x + b - y)²

---

### Derivative of MSE w.r.t. Weight (w)

```
L = (1/n) Σ (w·x_i + b - y_i)²

∂L/∂w = (1/n) Σ ∂/∂w [(w·x_i + b - y_i)²]
      = (1/n) Σ 2(w·x_i + b - y_i) · ∂/∂w (w·x_i + b - y_i)
      = (1/n) Σ 2(w·x_i + b - y_i) · x_i
      = (2/n) Σ (w·x_i + b - y_i) · x_i
      = (2/n) Σ (ŷ_i - y_i) · x_i
```

### Derivative of MSE w.r.t. Bias (b)

```
∂L/∂b = (1/n) Σ ∂/∂b [(w·x_i + b - y_i)²]
      = (1/n) Σ 2(w·x_i + b - y_i) · ∂/∂b (w·x_i + b - y_i)
      = (1/n) Σ 2(w·x_i + b - y_i) · 1
      = (2/n) Σ (w·x_i + b - y_i)
      = (2/n) Σ (ŷ_i - y_i)
```

---

### Gradient Descent Update Rule

```
w ← w - α · (∂L/∂w)
b ← b - α · (∂L/∂b)
```

**Why subtract?** The gradient points in the direction of steepest **increase**. We want to minimize, so we move in the opposite direction.

---

### Vectorized Form (NumPy)

```python
error = y_pred - y_true          # shape: (n,)
dL_dw = (2/n) * np.sum(error * X)
dL_db = (2/n) * np.sum(error)
```

---

### Key Insights

1. **MSE is convex for linear regression** → single global minimum
2. **Gradients → 0 at optimum** → stationary point
3. **Learning rate α controls step size**
   - Too large: overshoot, oscillate, diverge
   - Too small: slow convergence
4. **Final loss ≈ noise variance** (0.25 for σ=0.5) — cannot go lower
5. **Convergence verified**: w → 2.977 ≈ 3.0, b → 1.993 ≈ 2.0
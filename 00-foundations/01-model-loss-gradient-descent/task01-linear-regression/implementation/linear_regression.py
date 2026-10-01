"""
Linear Regression from Scratch (No Autograd, No Sklearn)
=========================================================
Implements the model/loss/gradient-descent loop by hand.

Task: S1-T1 — Linear Regression (By Hand)
- Toy dataset: y = 3x + 2 + noise
- Loss: Mean Squared Error (MSE)
- Gradients derived analytically: dL/dw, dL/db
- Gradient descent update: θ ← θ - α∇L

Expected: w → 3, b → 2
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. TOY DATASET GENERATION
# ============================================================
def generate_toy_data(n_samples: int = 100, true_w: float = 3.0, true_b: float = 2.0, noise_std: float = 0.5, seed: int = 42):
    """
    Generate y = true_w * x + true_b + noise
    Returns X (n_samples,), y (n_samples,)
    """
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = true_w * X + true_b + noise
    return X, y


# ============================================================
# 2. MODEL & LOSS (MSE)
# ============================================================
def forward(X: np.ndarray, w: float, b: float) -> np.ndarray:
    """Linear model: y_pred = w * X + b"""
    return w * X + b


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error: L = (1/n) * Σ(y_pred - y_true)²"""
    return np.mean((y_pred - y_true) ** 2)


# ============================================================
# 3. ANALYTICAL GRADIENTS (DERIVED BY HAND)
# ============================================================
# Derivation:
# L = (1/n) Σ (w*x_i + b - y_i)²
#
# dL/dw = (2/n) Σ (w*x_i + b - y_i) * x_i
# dL/db = (2/n) Σ (w*x_i + b - y_i)
#
# In vectorized form:
# dL/dw = (2/n) * X.T @ (y_pred - y_true)
# dL/db = (2/n) * Σ (y_pred - y_true)
def compute_gradients(X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """
    Compute gradients of MSE loss w.r.t. weight (w) and bias (b).
    Returns (dL_dw, dL_db)
    """
    n = len(X)
    error = y_pred - y_true  # (w*X + b) - y
    dL_dw = (2.0 / n) * np.sum(error * X)
    dL_db = (2.0 / n) * np.sum(error)
    return dL_dw, dL_db


# ============================================================
# 4. GRADIENT DESCENT TRAINING LOOP
# ============================================================
def train(X: np.ndarray, y: np.ndarray,
          w_init: float = 0.0, b_init: float = 0.0,
          lr: float = 0.01, epochs: int = 1000,
          verbose: bool = True, log_every: int = 100) -> tuple[float, float, list]:
    """
    Run gradient descent to minimize MSE.
    Returns (w_final, b_final, loss_history)
    """
    w, b = w_init, b_init
    loss_history = []

    for epoch in range(1, epochs + 1):
        # Forward pass
        y_pred = forward(X, w, b)

        # Loss
        loss = mse_loss(y_pred, y)
        loss_history.append(loss)

        # Gradients (analytical, by hand)
        dL_dw, dL_db = compute_gradients(X, y_pred, y)

        # Update: θ ← θ - α∇L  (subtract gradient!)
        w -= lr * dL_dw
        b -= lr * dL_db

        # Logging
        if verbose and epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | w: {w:.6f} | b: {b:.6f} | dL/dw: {dL_dw:.6f} | dL/db: {dL_db:.6f}")

    return w, b, loss_history


# ============================================================
# 5. VISUALIZATION
# ============================================================
def plot_results(X: np.ndarray, y: np.ndarray, w: float, b: float, loss_history: list,
                 true_w: float = 3.0, true_b: float = 2.0, save_path: str = None):
    """Plot data + fitted line + loss curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Data & fitted line
    ax = axes[0]
    ax.scatter(X, y, alpha=0.5, label='Data (y = 3x + 2 + noise)', s=20)
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = w * x_line + b
    y_true = true_w * x_line + true_b
    ax.plot(x_line, y_line, 'r-', label=f'Learned: w={w:.3f}, b={b:.3f}', linewidth=2)
    ax.plot(x_line, y_true, 'g--', label=f'True: w={true_w}, b={true_b}', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Loss curve
    ax = axes[1]
    ax.plot(loss_history, 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss Convergence')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == "__main__":
    # Hyperparameters
    LR = 0.01
    EPOCHS = 2000
    N_SAMPLES = 100
    TRUE_W, TRUE_B = 3.0, 2.0

    print("=" * 60)
    print("S1-T1: Linear Regression (By Hand)")
    print("=" * 60)
    print(f"True parameters: w={TRUE_W}, b={TRUE_B}")
    print(f"Hyperparameters: lr={LR}, epochs={EPOCHS}, n_samples={N_SAMPLES}")
    print("-" * 60)

    # Generate data
    X, y = generate_toy_data(n_samples=N_SAMPLES, true_w=TRUE_W, true_b=TRUE_B)

    # Train
    w_final, b_final, loss_history = train(
        X, y,
        w_init=0.0, b_init=0.0,
        lr=LR, epochs=EPOCHS,
        verbose=True, log_every=200
    )

    # Results
    print("-" * 60)
    print(f"Final: w={w_final:.6f}, b={b_final:.6f}")
    print(f"True:  w={TRUE_W:.6f}, b={TRUE_B:.6f}")
    print(f"Error: w_diff={abs(w_final - TRUE_W):.6f}, b_diff={abs(b_final - TRUE_B):.6f}")
    print(f"Final Loss: {loss_history[-1]:.6f}")

    # Verify convergence
    converged = abs(w_final - TRUE_W) < 0.1 and abs(b_final - TRUE_B) < 0.1
    print(f"Converged (within 0.1): {converged}")

    # Plot (non-interactive)
    exp_dir = Path(__file__).parent.parent / "experiments"
    plot_results(X, y, w_final, b_final, loss_history,
                 true_w=TRUE_W, true_b=TRUE_B,
                 save_path=exp_dir / "linear_regression_results.png")
    plt.close('all')

    # Save loss log
    log_path = exp_dir / "loss_log.txt"
    np.savetxt(log_path, loss_history, header="MSE Loss per epoch", fmt="%.6f")
    print(f"Loss log saved to {log_path}")

    print("=" * 60)
    print("Done!")
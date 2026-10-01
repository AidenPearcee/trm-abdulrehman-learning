"""
Unit Tests for Linear Regression Implementation
================================================
Run with: python -m pytest tests/ -v
"""

import sys
sys.path.insert(0, str(__file__).replace('tests/test_linear_regression.py', 'implementation'))

import numpy as np
from linear_regression import (
    generate_toy_data,
    forward,
    mse_loss,
    compute_gradients,
    train,
)


def test_forward():
    """Test forward pass computes w*x + b correctly."""
    X = np.array([1.0, 2.0, 3.0])
    w, b = 2.0, 1.0
    y_pred = forward(X, w, b)
    expected = np.array([3.0, 5.0, 7.0])
    assert np.allclose(y_pred, expected), f"Expected {expected}, got {y_pred}"
    print("✓ test_forward passed")


def test_mse_loss():
    """Test MSE loss computation."""
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 2.5])
    loss = mse_loss(y_pred, y_true)
    expected = np.mean((y_pred - y_true) ** 2)  # (0.25 + 0.25 + 0.25)/3 = 0.25
    assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"
    print("✓ test_mse_loss passed")


def test_compute_gradients():
    """Test gradient computation matches analytical derivation."""
    X = np.array([1.0, 2.0, 3.0])
    y_true = np.array([2.0, 4.0, 6.0])  # y = 2x, so true_w=2, true_b=0
    w, b = 1.0, 0.0  # Initial guess
    y_pred = forward(X, w, b)  # [1, 2, 3]
    
    dL_dw, dL_db = compute_gradients(X, y_pred, y_true)
    
    # Manual calculation:
    # error = [-1, -2, -3]
    # dL/dw = (2/3) * (-1*1 + -2*2 + -3*3) = (2/3) * (-1 -4 -9) = (2/3) * -14 = -28/3 ≈ -9.333
    # dL/db = (2/3) * (-1 -2 -3) = (2/3) * -6 = -4
    expected_dL_dw = (2/3) * (-14)
    expected_dL_db = (2/3) * (-6)
    
    assert np.isclose(dL_dw, expected_dL_dw), f"dL/dw: expected {expected_dL_dw}, got {dL_dw}"
    assert np.isclose(dL_db, expected_dL_db), f"dL/db: expected {expected_dL_db}, got {dL_db}"
    print("✓ test_compute_gradients passed")


def test_convergence():
    """Test that training converges close to true parameters."""
    np.random.seed(42)
    X, y = generate_toy_data(n_samples=100, true_w=3.0, true_b=2.0, noise_std=0.1)
    
    w_final, b_final, loss_history = train(
        X, y,
        w_init=0.0, b_init=0.0,
        lr=0.01, epochs=500,
        verbose=False
    )
    
    # Should converge within 0.1 of true values (with low noise)
    assert abs(w_final - 3.0) < 0.1, f"w={w_final} not close to 3.0"
    assert abs(b_final - 2.0) < 0.1, f"b={b_final} not close to 2.0"
    
    # Loss should decrease
    assert loss_history[-1] < loss_history[0], "Loss did not decrease"
    
    # Gradients should approach zero
    y_pred = forward(X, w_final, b_final)
    dL_dw, dL_db = compute_gradients(X, y_pred, y)
    assert abs(dL_dw) < 1e-3, f"dL/dw={dL_dw} not near zero"
    assert abs(dL_db) < 1e-3, f"dL/db={dL_db} not near zero"
    
    print("✓ test_convergence passed")


def test_learning_rate_effects():
    """Test that large learning rate causes divergence."""
    X, y = generate_toy_data(n_samples=50, true_w=3.0, true_b=2.0, noise_std=0.1)
    
    # Large LR should cause loss to increase (divergence)
    w_final, b_final, loss_history = train(
        X, y,
        w_init=0.0, b_init=0.0,
        lr=1.0, epochs=100,  # 100x larger than typical
        verbose=False
    )
    
    # Loss should increase or be NaN
    assert loss_history[-1] > loss_history[0] or np.isnan(loss_history[-1]), \
        "Large LR did not cause divergence"
    print("✓ test_learning_rate_effects passed")


if __name__ == "__main__":
    test_forward()
    test_mse_loss()
    test_compute_gradients()
    test_convergence()
    test_learning_rate_effects()
    print("\n✅ All tests passed!")
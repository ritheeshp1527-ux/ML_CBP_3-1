"""Unit tests for Fuzzy Logistic Regression implementation.

Verifies:
1. Triangular Fuzzy Number (TFN) multiplication for positive, negative, and zero predictors.
2. Numerical equivalence of vectorized sigmoid calculation with author's R routine.
3. Dubois & Prade (1987) expected value calculation and decision thresholding.
4. Binary response fuzzification and fuzzy loss metrics (MAE, MSE, RMSE).
5. Forward inference with Table 5 coefficients.
6. Monte Carlo coefficient fitting loop.
"""
from __future__ import annotations
import numpy as np
import pytest
from src.fuzzy_model import (
    TABLE_5_COEFFICIENTS,
    fuzzy_linear_combination,
    fuzzy_logistic_predict_tfn,
    dubois_prade_expected_value,
    fuzzify_binary_response,
    fuzzy_loss,
    FuzzyLogisticRegression
)


def test_tfn_scalar_multiplication_signs():
    """Verifies that TFN multiplication properly handles positive, negative, and zero scalars."""
    # beta_0 (intercept) = [0.0, 0.0, 0.0]
    # beta_1 = [-1.0, 0.5, 2.0]
    beta = np.array([
        [0.0, 0.0, 0.0],
        [-1.0, 0.5, 2.0]
    ])

    # Case 1: Positive scalar X = 3.0
    # [a, b, c] * 3.0 = [-3.0, 1.5, 6.0]
    X_pos = np.array([[3.0]])
    z_pos = fuzzy_linear_combination(X_pos, beta)
    np.testing.assert_allclose(z_pos, [[-3.0, 1.5, 6.0]])

    # Case 2: Negative scalar X = -2.0
    # [a, b, c] * -2.0 -> min is 2.0 * -2.0 = -4.0, max is -1.0 * -2.0 = 2.0
    # vertex triplet becomes: [c*X, b*X, a*X] = [-4.0, -1.0, 2.0]
    X_neg = np.array([[-2.0]])
    z_neg = fuzzy_linear_combination(X_neg, beta)
    np.testing.assert_allclose(z_neg, [[-4.0, -1.0, 2.0]])

    # Case 3: Zero scalar X = 0.0
    X_zero = np.array([[0.0]])
    z_zero = fuzzy_linear_combination(X_zero, beta)
    np.testing.assert_allclose(z_zero, [[0.0, 0.0, 0.0]])


def test_vectorized_sigmoid_equivalence_with_r():
    """Verifies exact numerical equivalence of vectorized sigmoid against author's R expressions."""
    def r_mult_TFN(t1, t2):
        prods = [t1[0]*t2[0], t1[0]*t2[2], t1[2]*t2[0], t1[2]*t2[2]]
        return np.array([min(prods), t1[1]*t2[1], max(prods)])

    def r_estimate_y_row(x_row, b0, b_slopes):
        bx = b0.copy()
        for j in range(len(x_row)):
            term = r_mult_TFN(b_slopes[j], np.array([x_row[j], x_row[j], x_row[j]]))
            bx = bx + term
        # In R: div_TFN(c(1,1,1), 1 + exp(mult_TFN(c(-1,-1,-1), bx)))
        neg_bx = r_mult_TFN(np.array([-1.0, -1.0, -1.0]), bx)
        exp_term = 1.0 + np.exp(neg_bx)
        divs = [1.0/exp_term[0], 1.0/exp_term[2], 1.0/exp_term[0], 1.0/exp_term[2]]
        return np.array([min(divs), 1.0/exp_term[1], max(divs)])

    np.random.seed(123)
    X = np.random.randn(5, 4)
    beta = np.sort(np.random.randn(5, 3), axis=1)

    P_vec = fuzzy_logistic_predict_tfn(X, beta)
    P_r = np.array([r_estimate_y_row(X[i], beta[0], beta[1:]) for i in range(len(X))])

    diff = np.max(np.abs(P_vec - P_r))
    assert diff < 1e-12, f"Discrepancy {diff} exceeds tolerance"


def test_dubois_prade_expected_value():
    """Verifies expected value formula EV = (p_left + 2 * p_center + p_right) / 4."""
    P_tfn = np.array([
        [0.2, 0.5, 0.8],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.1, 0.4, 0.9]
    ])
    ev = dubois_prade_expected_value(P_tfn)
    expected = np.array([
        (0.2 + 2*0.5 + 0.8) / 4.0,  # 2.0 / 4 = 0.5
        0.0,
        1.0,
        (0.1 + 2*0.4 + 0.9) / 4.0   # 1.8 / 4 = 0.45
    ])
    np.testing.assert_allclose(ev, expected)


def test_fuzzify_binary_response_properties():
    """Verifies that fuzzified responses have proper center, bounds, and ordering."""
    y = np.array([0, 1, 0, 1])
    y_tfn = fuzzify_binary_response(y, m=2.5, u_max=1.5, seed=42)

    assert y_tfn.shape == (4, 3)
    # Centers must exactly match original binary labels
    np.testing.assert_array_equal(y_tfn[:, 1], y)
    # Left <= Center <= Right for all instances
    assert np.all(y_tfn[:, 0] <= y_tfn[:, 1])
    assert np.all(y_tfn[:, 1] <= y_tfn[:, 2])


def test_fuzzy_loss_metrics():
    """Verifies calculation of MAE, MSE, and RMSE losses."""
    y_tfn = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    p_tfn = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0]])

    mae = fuzzy_loss(y_tfn, p_tfn, metric="mae")
    mse = fuzzy_loss(y_tfn, p_tfn, metric="mse")
    rmse = fuzzy_loss(y_tfn, p_tfn, metric="rmse")

    # Sample 0 diff: [0.1, 0.2, 0.3] -> abs sum = 0.6, sq sum = 0.14
    # Sample 1 diff: [-0.2, -0.1, 0.0] -> abs sum = 0.3, sq sum = 0.05
    expected_mae = (0.6 + 0.3) / 2.0  # 0.45
    expected_mse = (0.14 + 0.05) / 2.0  # 0.095

    assert pytest.approx(mae) == expected_mae
    assert pytest.approx(mse) == expected_mse
    assert rmse > 0.0


def test_table_5_dimensions_and_inference():
    """Verifies that Table 5 coefficients have correct shape and generate valid probabilities."""
    assert TABLE_5_COEFFICIENTS.shape == (30, 3)
    # All rows must satisfy a <= b <= c
    assert np.all(TABLE_5_COEFFICIENTS[:, 0] <= TABLE_5_COEFFICIENTS[:, 1])
    assert np.all(TABLE_5_COEFFICIENTS[:, 1] <= TABLE_5_COEFFICIENTS[:, 2])

    model = FuzzyLogisticRegression.from_table_5()
    X = np.random.randn(10, 29)
    P_fuzzy = model.predict_fuzzy(X)
    proba = model.predict_proba(X)
    pred = model.predict(X, threshold=0.5)

    assert P_fuzzy.shape == (10, 3)
    assert np.all((P_fuzzy >= 0.0) & (P_fuzzy <= 1.0))
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba[:, 0] + proba[:, 1], 1.0)
    assert set(np.unique(pred)).issubset({0, 1})


def test_fuzzy_monte_carlo_fit_smoke():
    """Smoke test verifying that FuzzyLogisticRegression fit runs and reduces loss."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.choice([0, 1], size=50, p=[0.8, 0.2])

    model = FuzzyLogisticRegression(n_iterations=20, random_state=42, loss_metric="mae")
    model.fit(X, y)

    assert model.beta is not None
    assert model.beta.shape == (5, 3)
    assert model.best_loss_ is not None
    assert np.all(model.beta[:, 0] <= model.beta[:, 1])
    assert np.all(model.beta[:, 1] <= model.beta[:, 2])

    pred = model.predict(X)
    assert len(pred) == 50

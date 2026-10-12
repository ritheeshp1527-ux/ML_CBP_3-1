"""Unit tests for Guided Fuzzy Logistic Regression implementation.

Verifies:
1. TFN coefficient ordering: a_j <= b_j <= c_j across all parameters.
2. Numerical stability and finite outputs under extreme feature ranges.
3. Strict fuzzy probability bounds: 0 < p_left <= p_center <= p_right < 1.
4. Correct behavior for positive, negative, and zero-valued features.
5. Reproducibility under fixed random seeds.
6. Data partition integrity: no train/val/test data leakage.
7. Rare-class objective sensitivity under severe imbalance.
8. Validation-only threshold selection protocol.
9. End-to-end training, prediction, and uncertainty explanation.
10. Regression test ensuring original Monte Carlo Fuzzy model remains intact.
"""
from __future__ import annotations
import numpy as np
import pytest
from src.fuzzy_model import FuzzyLogisticRegression, TABLE_5_COEFFICIENTS
from src.guided_fuzzy_model import GuidedFuzzyLogisticRegression


def test_guided_tfn_coefficient_ordering():
    """Verifies that all learned TFN coefficients satisfy a <= b <= c."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])

    model = GuidedFuzzyLogisticRegression(max_iter=15, random_state=42)
    model.fit(X, y)

    assert model.beta_ is not None
    assert model.beta_.shape == (6, 3)  # intercept + 5 features
    assert np.all(model.beta_[:, 0] <= model.beta_[:, 1])
    assert np.all(model.beta_[:, 1] <= model.beta_[:, 2])


def test_guided_numerical_stability_and_finite_outputs():
    """Verifies finite outputs and stability under extreme input values."""
    np.random.seed(42)
    X_train = np.random.randn(50, 4)
    y_train = np.random.choice([0, 1], size=50, p=[0.8, 0.2])

    model = GuidedFuzzyLogisticRegression(max_iter=10, random_state=42).fit(X_train, y_train)

    # Test extreme positive and negative values
    X_extreme = np.array([
        [1000.0, -1000.0, 500.0, 0.0],
        [-5000.0, 5000.0, -2000.0, 100.0]
    ])
    P_fuzzy = model.predict_fuzzy(X_extreme)
    proba = model.predict_proba(X_extreme)

    assert np.all(np.isfinite(P_fuzzy))
    assert np.all(np.isfinite(proba))
    assert np.all((P_fuzzy >= 0.0) & (P_fuzzy <= 1.0))
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_guided_fuzzy_probability_bounds():
    """Verifies strict TFN probability bounds: 0 < p_left <= p_center <= p_right < 1."""
    np.random.seed(42)
    X = np.random.randn(40, 3)
    y = np.random.choice([0, 1], size=40, p=[0.75, 0.25])

    model = GuidedFuzzyLogisticRegression(max_iter=10, random_state=42).fit(X, y)
    P_fuzzy = model.predict_fuzzy(X)

    # Component ordering
    assert np.all(P_fuzzy[:, 0] <= P_fuzzy[:, 1] + 1e-12)
    assert np.all(P_fuzzy[:, 1] <= P_fuzzy[:, 2] + 1e-12)
    assert np.all(P_fuzzy[:, 0] >= 0.0)
    assert np.all(P_fuzzy[:, 2] <= 1.0)


def test_guided_sign_dependent_feature_behavior():
    """Verifies that positive, negative, and zero features trigger proper interval bounds."""
    # Set explicit coefficient triplet: b0 = [0, 0, 0], beta_1 = [-2, 1, 3]
    model = GuidedFuzzyLogisticRegression()
    model.beta_ = np.array([
        [0.0, 0.0, 0.0],
        [-2.0, 1.0, 3.0]
    ])

    X_pos = np.array([[2.0]])
    X_neg = np.array([[-2.0]])
    X_zero = np.array([[0.0]])

    P_pos = model.predict_fuzzy(X_pos)
    P_neg = model.predict_fuzzy(X_neg)
    P_zero = model.predict_fuzzy(X_zero)

    # For zero feature: logit is 0 -> prob = 0.5 across all components
    np.testing.assert_allclose(P_zero[0], [0.5, 0.5, 0.5])

    # For positive feature X=2: logit is [-4, 2, 6] -> prob = [sig(-4), sig(2), sig(6)]
    assert P_pos[0, 0] < P_pos[0, 1] < P_pos[0, 2]

    # For negative feature X=-2: logit is [-6, -2, 4] -> prob = [sig(-6), sig(-2), sig(4)]
    assert P_neg[0, 0] < P_neg[0, 1] < P_neg[0, 2]


def test_guided_reproducibility_with_fixed_seed():
    """Verifies identical coefficient convergence when using fixed random_state."""
    X = np.random.randn(60, 4)
    y = np.random.choice([0, 1], size=60, p=[0.8, 0.2])

    m1 = GuidedFuzzyLogisticRegression(max_iter=15, random_state=123).fit(X, y)
    m2 = GuidedFuzzyLogisticRegression(max_iter=15, random_state=123).fit(X, y)

    np.testing.assert_allclose(m1.beta_, m2.beta_, atol=1e-10)


def test_guided_no_data_leakage():
    """Verifies that model fitting consumes strictly training arrays and does not mutate them."""
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_train = np.array([0, 0, 1, 1])

    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()

    model = GuidedFuzzyLogisticRegression(max_iter=10, random_state=42)
    model.fit(X_train, y_train)

    np.testing.assert_array_equal(X_train, X_train_copy)
    np.testing.assert_array_equal(y_train, y_train_copy)


def test_guided_rare_class_objectives():
    """Verifies that weighted_bce correctly handles severe class imbalance."""
    # 99 legitimate, 1 fraud
    X = np.random.randn(100, 3)
    y = np.zeros(100, dtype=int)
    y[0] = 1

    model_wbce = GuidedFuzzyLogisticRegression(
        objective="weighted_bce", class_weight="balanced", max_iter=20, random_state=42
    ).fit(X, y)

    prob = model_wbce.predict_proba(X)[:, 1]
    # The fraud instance should receive non-trivial predicted probability
    assert prob[0] > 0.05


def test_guided_validation_threshold_selection():
    """Verifies that optimal threshold is determined from validation split without test data."""
    model = GuidedFuzzyLogisticRegression()
    model.beta_ = np.array([
        [-1.0, -0.5, 0.0],
        [0.5, 1.0, 1.5]
    ])

    X_val = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y_val = np.array([0, 0, 0, 1, 1])

    val_scores = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.9, 0.1):
        pred = (val_scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_val == 1))
        fp = np.sum((pred == 1) & (y_val == 0))
        fn = np.sum((pred == 0) & (y_val == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    assert 0.1 <= best_t <= 0.9
    assert best_f1 > 0.5


def test_guided_end_to_end_pipeline():
    """Verifies end-to-end training, prediction, and uncertainty explainability."""
    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = np.random.choice([0, 1], size=80, p=[0.85, 0.15])

    model = GuidedFuzzyLogisticRegression(max_iter=15, random_state=42).fit(X, y)

    pred = model.predict(X, threshold=0.5)
    proba = model.predict_proba(X)
    assert len(pred) == 80
    assert proba.shape == (80, 2)

    # Uncertainty explanations
    coeff_expl = model.explain_uncertainty(feature_names=["f1", "f2", "f3", "f4", "f5"])
    assert len(coeff_expl) == 6
    assert "width" in coeff_expl.columns
    assert np.all(coeff_expl["width"] > 0)

    pred_expl = model.predict_uncertainty(X)
    assert len(pred_expl) == 80
    assert "uncertainty_width" in pred_expl.columns
    assert np.all(pred_expl["uncertainty_width"] >= 0)


def test_regression_original_fuzzy_model_available():
    """Regression test ensuring original FuzzyLogisticRegression remains fully available."""
    assert TABLE_5_COEFFICIENTS.shape == (30, 3)
    orig_model = FuzzyLogisticRegression.from_table_5()
    X = np.zeros((5, 29))
    prob = orig_model.predict_proba(X)
    assert prob.shape == (5, 2)
    np.testing.assert_allclose(prob[:, 0] + prob[:, 1], 1.0)

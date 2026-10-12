"""Fuzzy Logistic Regression implementation for credit card fraud detection.

Based on:
Charizanos, G., Demirhan, H., & Icen, D. (2024).
An online fuzzy fraud detection framework for credit card transactions.
Expert Systems with Applications, 252, 124127.
"""
from __future__ import annotations
import numpy as np
from typing import Literal

# Published Table 5 Triangular Fuzzy Number (TFN) coefficients (30 total: intercept + 29 features).
# Format: [left_vertex (a), center_vertex (b), right_vertex (c)]
TABLE_5_COEFFICIENTS = np.array([
    [-0.920, -0.587,  0.552],  # beta_0: Intercept
    [-0.699, -0.458, -0.314],  # beta_1: V1
    [-0.963,  0.345,  0.638],  # beta_2: V2
    [ 0.143,  0.316,  0.503],  # beta_3: V3
    [-0.889, -0.518,  0.664],  # beta_4: V4
    [-0.611, -0.504,  0.256],  # beta_5: V5
    [-0.229,  0.105,  0.837],  # beta_6: V6
    [ 0.017,  0.397,  0.597],  # beta_7: V7
    [-0.455,  0.223,  0.778],  # beta_8: V8
    [-0.533, -0.071,  0.895],  # beta_9: V9
    [-0.954, -0.654,  0.047],  # beta_10: V10
    [-0.780, -0.169,  0.693],  # beta_11: V11
    [ 0.248,  0.477,  0.984],  # beta_12: V12
    [-0.073,  0.733,  0.758],  # beta_13: V13
    [-0.456, -0.301,  0.100],  # beta_14: V14
    [-0.082,  0.433,  0.529],  # beta_15: V15
    [ 0.199,  0.482,  0.854],  # beta_16: V16
    [-0.671, -0.095,  0.184],  # beta_17: V17
    [-0.913, -0.443,  0.047],  # beta_18: V18
    [-0.932,  0.068,  0.493],  # beta_19: V19
    [-0.955, -0.336,  0.161],  # beta_20: V20
    [-0.605, -0.254, -0.018],  # beta_21: V21
    [-0.591, -0.534,  0.582],  # beta_22: V22
    [-0.651, -0.366,  0.508],  # beta_23: V23
    [-0.260,  0.363,  0.425],  # beta_24: V24
    [-0.836, -0.820, -0.672],  # beta_25: V25
    [ 0.289,  0.581,  0.931],  # beta_26: V26
    [-0.520, -0.176,  0.259],  # beta_27: V27
    [-0.162,  0.140,  0.319],  # beta_28: V28
    [-0.157, -0.066,  0.009],  # beta_29: V29 (Amount)
], dtype=np.float64)


def fuzzy_linear_combination(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Computes linear combination z = beta_0 + sum(beta_j * X_j) using TFN arithmetic.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Crisp predictor matrix.
    beta : np.ndarray of shape (n_features + 1, 3)
        TFN coefficients [a, b, c] with a <= b <= c.
        beta[0] is the intercept; beta[1:] are feature slopes.
        
    Returns
    -------
    Z : np.ndarray of shape (n_samples, 3)
        Fuzzy logit triplets [z_left, z_center, z_right].
    """
    X = np.asarray(X, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape[0] != X.shape[1] + 1:
        raise ValueError(
            f"Coefficient length {beta.shape[0]} does not match features {X.shape[1]} + 1 (intercept)."
        )
    b0 = beta[0]
    b = beta[1:]
    a, b_ctr, c = b[:, 0], b[:, 1], b[:, 2]

    # For positive X_j, [a, b, c] * X_j = [a*X_j, b*X_j, c*X_j]
    # For negative X_j, [a, b, c] * X_j = [c*X_j, b*X_j, a*X_j]
    X_pos = np.maximum(X, 0.0)
    X_neg = np.minimum(X, 0.0)

    z_left = b0[0] + X_pos @ a + X_neg @ c
    z_center = b0[1] + X @ b_ctr
    z_right = b0[2] + X_pos @ c + X_neg @ a

    return np.column_stack([z_left, z_center, z_right])


def fuzzy_logistic_predict_tfn(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Evaluates the fuzzy logistic function P = 1 / (1 + exp(-Z)) component-wise.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    beta : np.ndarray of shape (n_features + 1, 3)
    
    Returns
    -------
    P : np.ndarray of shape (n_samples, 3)
        Fuzzy probability triplets [p_left, p_center, p_right] in (0, 1).
    """
    Z = fuzzy_linear_combination(X, beta)
    # Component-wise sigmoid with clipping to prevent overflow
    Z_clipped = np.clip(Z, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-Z_clipped))


def dubois_prade_expected_value(P_tfn: np.ndarray) -> np.ndarray:
    """
    Computes Dubois & Prade (1987) expected value for TFNs:
    EV = (p_left + 2 * p_center + p_right) / 4
    
    Parameters
    ----------
    P_tfn : np.ndarray of shape (n_samples, 3)
    
    Returns
    -------
    ev : np.ndarray of shape (n_samples,)
    """
    P_tfn = np.asarray(P_tfn, dtype=np.float64)
    return (P_tfn[:, 0] + 2.0 * P_tfn[:, 1] + P_tfn[:, 2]) / 4.0


def fuzzify_binary_response(
    y: np.ndarray,
    m: float = 2.5,
    l: float = 1.0,
    r: float = 1.0,
    u_min: float = 0.0,
    u_max: float = 1.5,
    seed: int | None = 123
) -> np.ndarray:
    """
    Fuzzifies binary response Y in {0, 1} into TFNs [left, center, right]
    following author's fuzzify_Y function.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    center = y.astype(np.float64)
    left = np.empty(n, dtype=np.float64)
    right = np.empty(n, dtype=np.float64)

    mask0 = (y == 0)
    mask1 = (y == 1)

    u0_l = rng.uniform(u_min, u_max, size=np.sum(mask0))
    u0_r = rng.uniform(u_min, u_max, size=np.sum(mask0))
    left[mask0] = 0.0 - l * m * u0_l
    right[mask0] = 0.0 + r * m * u0_r

    u1_l = rng.uniform(u_min, u_max, size=np.sum(mask1))
    u1_r = rng.uniform(u_min, u_max, size=np.sum(mask1))
    left[mask1] = 1.0 - l * m * u1_l
    right[mask1] = 1.0 + r * m * u1_r

    return np.column_stack([left, center, right])


def fuzzy_loss(
    y_tfn: np.ndarray,
    p_tfn: np.ndarray,
    metric: Literal["mae", "mse", "rmse"] = "mae"
) -> float:
    """
    Computes fuzzy loss between true fuzzified response y_tfn and predicted p_tfn
    following the definitions in Charizanos et al. (2024) and their R implementation.
    """
    diff = y_tfn - p_tfn
    if metric == "mae":
        # Sum of absolute differences across 3 vertices, averaged over samples
        return float(np.mean(np.sum(np.abs(diff), axis=1)))
    elif metric == "mse":
        # Sum of squared differences across 3 vertices, averaged over samples
        return float(np.mean(np.sum(diff ** 2, axis=1)))
    elif metric == "rmse":
        # Average RMSE per sample: mean(sqrt(mean((p - actual)^2)))
        sample_rmse = np.sqrt(np.mean(diff ** 2, axis=1))
        return float(np.mean(sample_rmse))
    else:
        raise ValueError(f"Unknown metric {metric}")


class FuzzyLogisticRegression:
    """
    Fuzzy Logistic Regression Classifier (Case II: crisp predictors, fuzzy coefficients).
    
    Supports:
    - Pre-trained coefficient loading (e.g. Table 5 published weights).
    - Monte Carlo parameter optimization matching author's learning stage.
    """
    def __init__(
        self,
        beta: np.ndarray | None = None,
        n_iterations: int = 1000,
        random_state: int | None = 42,
        b0_lower: float = -1.0,
        b0_upper: float = 1.0,
        bi_lower: float = -1.0,
        bi_upper: float = 1.0,
        loss_metric: Literal["mae", "mse", "rmse"] = "mae",
        fuzzification_factor: float = 2.5,
        fuzzy_y_max: float = 1.5,
    ):
        self.beta = np.copy(beta) if beta is not None else None
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.b0_lower = b0_lower
        self.b0_upper = b0_upper
        self.bi_lower = bi_lower
        self.bi_upper = bi_upper
        self.loss_metric = loss_metric
        self.fuzzification_factor = fuzzification_factor
        self.fuzzy_y_max = fuzzy_y_max
        self.best_loss_ = None

    @classmethod
    def from_table_5(cls) -> "FuzzyLogisticRegression":
        """Instantiate with the author's published Table 5 coefficients."""
        return cls(beta=TABLE_5_COEFFICIENTS)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzyLogisticRegression":
        """
        Fits fuzzy coefficients using author's Monte Carlo random search procedure
        on training data (Case II in Charizanos et al.).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Fuzzify training labels using author configuration
        y_tfn = fuzzify_binary_response(
            y,
            m=self.fuzzification_factor,
            u_max=self.fuzzy_y_max,
            seed=self.random_state
        )

        best_beta = None
        best_loss = float("inf")

        # Monte Carlo search across coefficient space
        for _ in range(self.n_iterations):
            # Draw random sorted TFN for intercept b0
            b0_draw = np.sort(rng.uniform(self.b0_lower, self.b0_upper, size=3))
            # Draw random sorted TFNs for feature slopes
            bi_draw = np.sort(
                rng.uniform(self.bi_lower, self.bi_upper, size=(n_features, 3)),
                axis=1
            )
            candidate_beta = np.vstack([b0_draw, bi_draw])

            # Predict and compute loss
            p_tfn = fuzzy_logistic_predict_tfn(X, candidate_beta)
            loss = fuzzy_loss(y_tfn, p_tfn, metric=self.loss_metric)

            if loss < best_loss:
                best_loss = loss
                best_beta = candidate_beta

        self.beta = best_beta
        self.best_loss_ = best_loss
        return self

    def predict_fuzzy(self, X: np.ndarray) -> np.ndarray:
        """Returns fuzzy probability triplets [p_left, p_center, p_right]."""
        if self.beta is None:
            raise ValueError("Model is not fitted and no coefficients provided.")
        return fuzzy_logistic_predict_tfn(X, self.beta)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns crisp probabilities using Dubois & Prade expected value:
        P(Class=1) = E(P_tfn)
        """
        P_tfn = self.predict_fuzzy(X)
        ev = dubois_prade_expected_value(P_tfn)
        # Returns [P(0), P(1)] for scikit-learn compatibility
        return np.column_stack([1.0 - ev, ev])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions based on expected value threshold."""
        prob1 = self.predict_proba(X)[:, 1]
        return (prob1 >= threshold).astype(int)

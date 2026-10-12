"""Guided Fuzzy Logistic Regression Model.

Combines discriminative parameter initialization (from training-partition logistic regression)
with bounded optimization to learn valid Triangular Fuzzy Number (TFN) coefficients
[a_j <= b_j <= c_j] that preserve fuzzy arithmetic and uncertainty explainability while
addressing the class imbalance and non-convexity challenges of credit card fraud detection.

Based on:
Charizanos, G., Demirhan, H., & Icen, D. (2024).
An online fuzzy fraud detection framework for credit card transactions.
Expert Systems with Applications, 252, 124127.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Literal

from .fuzzy_model import (
    fuzzy_linear_combination,
    fuzzy_logistic_predict_tfn,
    dubois_prade_expected_value,
    fuzzify_binary_response,
)


class GuidedFuzzyLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Guided Fuzzy Logistic Regression Classifier.
    
    Learns valid Triangular Fuzzy Number (TFN) coefficients beta_j = [a_j, b_j, c_j]
    such that a_j <= b_j <= c_j for all j = 0..D.
    
    Parameters
    ----------
    objective : {'weighted_bce', 'class_weighted_fuzzy_mae', 'unweighted_fuzzy_mae'}
        The learning objective function.
    class_weight : 'balanced' or None, default='balanced'
        Weighting scheme used for initialization and weighted loss objectives.
    spread_init : {'fisher_se', 'fixed'}, default='fisher_se'
        Strategy to initialize coefficient spreads.
    spread_factor : float, default=1.0
        Multiplier kappa for initial spreads (e.g. kappa * SE).
    spread_reg : float, default=0.005
        L1 regularization coefficient lambda on spreads to prevent over-expansion.
    max_iter : int, default=60
        Maximum iterations for L-BFGS-B bounded optimization.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    def __init__(
        self,
        objective: Literal["weighted_bce", "class_weighted_fuzzy_mae", "unweighted_fuzzy_mae"] = "weighted_bce",
        class_weight: str | None = "balanced",
        spread_init: Literal["fisher_se", "fixed"] = "fisher_se",
        spread_factor: float = 1.0,
        spread_reg: float = 0.005,
        max_iter: int = 60,
        random_state: int = 42,
    ):
        self.objective = objective
        self.class_weight = class_weight
        self.spread_init = spread_init
        self.spread_factor = spread_factor
        self.spread_reg = spread_reg
        self.max_iter = max_iter
        self.random_state = random_state

        # Attributes set after fitting
        self.beta_: np.ndarray | None = None
        self.init_beta_: np.ndarray | None = None
        self.training_loss_: float | None = None
        self.n_iter_: int | None = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GuidedFuzzyLogisticRegression":
        """
        Fits guided TFN coefficients strictly using training data.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X.shape

        if set(np.unique(y)) - {0, 1}:
            raise ValueError("y must contain binary labels in {0, 1}.")

        # 1. Reference Center Initialization via Training-Partition Logistic Regression
        lr_ref = LogisticRegression(
            class_weight=self.class_weight,
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs"
        )
        lr_ref.fit(X, y)
        b0_init = float(lr_ref.intercept_[0])
        b_init = lr_ref.coef_[0].copy()

        # 2. Principled Spread Initialization via Fisher Information Covariance
        p_init = 1.0 / (1.0 + np.exp(-np.clip(b0_init + X @ b_init, -30.0, 30.0)))
        w_diag = p_init * (1.0 - p_init)
        X_bias = np.column_stack([np.ones(n_samples), X])
        # Regularized Fisher information matrix
        info_mat = X_bias.T @ (w_diag[:, None] * X_bias) + 1.0 * np.eye(n_features + 1)
        cov_mat = np.linalg.inv(info_mat)
        se = np.sqrt(np.maximum(np.diag(cov_mat), 1e-4))

        if self.spread_init == "fisher_se":
            sl0_init = float(self.spread_factor * se[0])
            sl_init = (self.spread_factor * se[1:]).astype(np.float64)
            sr0_init = sl0_init
            sr_init = sl_init.copy()
        else:
            sl0_init = float(0.1 * self.spread_factor)
            sl_init = np.full(n_features, 0.1 * self.spread_factor, dtype=np.float64)
            sr0_init = sl0_init
            sr_init = sl_init.copy()

        # Record initial TFN coefficients
        init_a0, init_c0 = b0_init - sl0_init, b0_init + sr0_init
        init_a = b_init - sl_init
        init_c = b_init + sr_init
        self.init_beta_ = np.vstack([
            [init_a0, b0_init, init_c0],
            np.column_stack([init_a, b_init, init_c])
        ])

        # Parameter vector packing:
        # [b0, b1..bD, sl0, sl1..slD, sr0, sr1..srD] -> length 3*(D+1)
        init_params = np.concatenate([
            [b0_init], b_init,
            [sl0_init], sl_init,
            [sr0_init], sr_init
        ])

        # Bounded constraints: center unbounded, spreads strictly positive [0.005, 5.0]
        bounds = [(None, None)] * (n_features + 1) + [(0.005, 5.0)] * (2 * (n_features + 1))

        # Class weights for loss
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        w_pos = float(n_samples / (2.0 * max(n_pos, 1)))
        w_neg = float(n_samples / (2.0 * max(n_neg, 1)))

        # Pre-compute positive and negative feature splits for vectorized linear combination
        X_pos = np.maximum(X, 0.0)
        X_neg = np.minimum(X, 0.0)

        # Fuzzified target response for MAE objectives
        y_tfn = fuzzify_binary_response(y, m=2.5, u_max=1.5, seed=self.random_state)

        def _forward_tfn(params: np.ndarray):
            b0 = params[0]
            b = params[1:n_features + 1]
            sl0 = params[n_features + 1]
            sl = params[n_features + 2:2 * n_features + 2]
            sr0 = params[2 * n_features + 2]
            sr = params[2 * n_features + 3:]

            a0, c0 = b0 - sl0, b0 + sr0
            a, c = b - sl, b + sr

            zl = a0 + X_pos @ a + X_neg @ c
            zc = b0 + X @ b
            zr = c0 + X_pos @ c + X_neg @ a

            pl = 1.0 / (1.0 + np.exp(-np.clip(zl, -50.0, 50.0)))
            pc = 1.0 / (1.0 + np.exp(-np.clip(zc, -50.0, 50.0)))
            pr = 1.0 / (1.0 + np.exp(-np.clip(zr, -50.0, 50.0)))
            P_triplet = np.column_stack([pl, pc, pr])
            ev = (pl + 2.0 * pc + pr) / 4.0
            return P_triplet, ev

        def _objective_func(params: np.ndarray) -> float:
            P_triplet, ev = _forward_tfn(params)
            spread_sum = np.sum(params[n_features + 1:])

            if self.objective == "weighted_bce":
                ev_safe = np.clip(ev, 1e-12, 1.0 - 1e-12)
                loss = -np.mean(w_pos * y * np.log(ev_safe) + w_neg * (1 - y) * np.log(1.0 - ev_safe))
                return float(loss + self.spread_reg * spread_sum)

            elif self.objective == "class_weighted_fuzzy_mae":
                diff = np.abs(y_tfn - P_triplet)
                sample_mae = np.sum(diff, axis=1)
                weights = np.where(y == 1, w_pos, w_neg)
                loss = np.mean(weights * sample_mae)
                return float(loss + self.spread_reg * spread_sum)

            elif self.objective == "unweighted_fuzzy_mae":
                diff = np.abs(y_tfn - P_triplet)
                sample_mae = np.sum(diff, axis=1)
                loss = np.mean(sample_mae)
                return float(loss + self.spread_reg * spread_sum)

            else:
                raise ValueError(f"Unknown objective {self.objective}")

        # Execute bounded L-BFGS-B optimization
        opt_res = minimize(
            _objective_func,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter}
        )

        # Unpack final learned parameters
        opt_p = opt_res.x
        b0_f = opt_p[0]
        b_f = opt_p[1:n_features + 1]
        sl0_f = opt_p[n_features + 1]
        sl_f = opt_p[n_features + 2:2 * n_features + 2]
        sr0_f = opt_p[2 * n_features + 2]
        sr_f = opt_p[2 * n_features + 3:]

        a0_f, c0_f = b0_f - sl0_f, b0_f + sr0_f
        a_f = b_f - sl_f
        c_f = b_f + sr_f

        self.beta_ = np.vstack([
            [a0_f, b0_f, c0_f],
            np.column_stack([a_f, b_f, c_f])
        ])
        self.spreads_left_ = np.concatenate([[sl0_f], sl_f])
        self.spreads_right_ = np.concatenate([[sr0_f], sr_f])
        self.training_loss_ = float(opt_res.fun)
        self.n_iter_ = int(opt_res.nit)

        return self

    def update_batch(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        replay_X: np.ndarray | None = None,
        replay_y: np.ndarray | None = None,
        proximal_weight: float = 1.0,
        max_iter: int = 10,
    ) -> float:
        """
        Incrementally updates guided TFN coefficients on newly arrived labeled batch.
        
        Preserves strict TFN ordering [a_j <= b_j <= c_j] through bounded spread constraints.
        Uses proximal regularization towards current parameters to prevent catastrophic
        forgetting under severe class imbalance.
        """
        if self.beta_ is None:
            raise ValueError("Model must be fitted before online update.")
        
        X_batch = np.asarray(X_batch, dtype=np.float64)
        y_batch = np.asarray(y_batch, dtype=np.int64)
        
        if replay_X is not None and replay_y is not None and len(replay_X) > 0:
            X_all = np.vstack([X_batch, np.asarray(replay_X, dtype=np.float64)])
            y_all = np.concatenate([y_batch, np.asarray(replay_y, dtype=np.int64)])
        else:
            X_all = X_batch
            y_all = y_batch
            
        n_samples, n_features = X_all.shape
        if n_features != len(self.beta_) - 1:
            raise ValueError(f"Feature count mismatch: expected {len(self.beta_) - 1}, got {n_features}")
            
        n_pos = np.sum(y_all == 1)
        n_neg = np.sum(y_all == 0)
        w_pos = float(n_samples / (2.0 * max(n_pos, 1)))
        w_neg = float(n_samples / (2.0 * max(n_neg, 1)))
        
        b0_curr = self.beta_[0, 1]
        b_curr = self.beta_[1:, 1]
        sl0_curr = self.spreads_left_[0]
        sl_curr = self.spreads_left_[1:]
        sr0_curr = self.spreads_right_[0]
        sr_curr = self.spreads_right_[1:]
        
        init_params = np.concatenate([
            [b0_curr], b_curr,
            [sl0_curr], sl_curr,
            [sr0_curr], sr_curr
        ])
        
        bounds = [(None, None)] * (n_features + 1) + [(0.005, 5.0)] * (2 * (n_features + 1))
        
        X_pos = np.maximum(X_all, 0.0)
        X_neg = np.minimum(X_all, 0.0)
        
        def _obj(params: np.ndarray) -> float:
            b0 = params[0]
            b = params[1:n_features + 1]
            sl0 = params[n_features + 1]
            sl = params[n_features + 2:2 * n_features + 2]
            sr0 = params[2 * n_features + 2]
            sr = params[2 * n_features + 3:]

            a0, c0 = b0 - sl0, b0 + sr0
            a, c = b - sl, b + sr

            zl = a0 + X_pos @ a + X_neg @ c
            zc = b0 + X_all @ b
            zr = c0 + X_pos @ c + X_neg @ a

            pl = 1.0 / (1.0 + np.exp(-np.clip(zl, -50.0, 50.0)))
            pc = 1.0 / (1.0 + np.exp(-np.clip(zc, -50.0, 50.0)))
            pr = 1.0 / (1.0 + np.exp(-np.clip(zr, -50.0, 50.0)))
            ev = (pl + 2.0 * pc + pr) / 4.0
            ev_safe = np.clip(ev, 1e-12, 1.0 - 1e-12)
            
            bce = -np.mean(w_pos * y_all * np.log(ev_safe) + w_neg * (1 - y_all) * np.log(1.0 - ev_safe))
            spread_sum = np.sum(params[n_features + 1:])
            prox_penalty = 0.5 * proximal_weight * np.sum((params - init_params) ** 2)
            return float(bce + self.spread_reg * spread_sum + prox_penalty)
            
        opt_res = minimize(
            _obj,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter}
        )
        
        opt_p = opt_res.x
        b0_f = opt_p[0]
        b_f = opt_p[1:n_features + 1]
        sl0_f = opt_p[n_features + 1]
        sl_f = opt_p[n_features + 2:2 * n_features + 2]
        sr0_f = opt_p[2 * n_features + 2]
        sr_f = opt_p[2 * n_features + 3:]

        a0_f, c0_f = b0_f - sl0_f, b0_f + sr0_f
        a_f = b_f - sl_f
        c_f = b_f + sr_f

        self.beta_ = np.vstack([
            [a0_f, b0_f, c0_f],
            np.column_stack([a_f, b_f, c_f])
        ])
        self.spreads_left_ = np.concatenate([[sl0_f], sl_f])
        self.spreads_right_ = np.concatenate([[sr0_f], sr_f])
        return float(opt_res.fun)


    def predict_fuzzy(self, X: np.ndarray) -> np.ndarray:
        """
        Returns fuzzy probability triplets [p_left, p_center, p_right].
        Strictly satisfies 0 < p_left <= p_center <= p_right < 1.
        """
        if self.beta_ is None:
            raise ValueError("Model is not fitted.")
        return fuzzy_logistic_predict_tfn(X, self.beta_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns crisp probabilities using Dubois & Prade expected value:
        P(Class=1) = E(P_tfn)
        """
        P_tfn = self.predict_fuzzy(X)
        ev = dubois_prade_expected_value(P_tfn)
        return np.column_stack([1.0 - ev, ev])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary fraud classification based on thresholded expected value."""
        prob_fraud = self.predict_proba(X)[:, 1]
        return (prob_fraud >= threshold).astype(int)

    def explain_uncertainty(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        """
        Returns a DataFrame explaining TFN coefficients, vertex widths, and asymmetry.
        """
        if self.beta_ is None:
            raise ValueError("Model is not fitted.")
        if feature_names is None:
            feature_names = ["Intercept"] + [f"V{i}" for i in range(1, len(self.beta_))]
        else:
            feature_names = ["Intercept"] + list(feature_names)

        df = pd.DataFrame({
            "feature": feature_names,
            "tfn_a_lower": self.beta_[:, 0],
            "tfn_b_center": self.beta_[:, 1],
            "tfn_c_upper": self.beta_[:, 2],
            "width": self.beta_[:, 2] - self.beta_[:, 0],
            "left_spread": self.spreads_left_,
            "right_spread": self.spreads_right_,
            "asymmetry_ratio": np.where(
                (self.beta_[:, 2] - self.beta_[:, 0]) > 0,
                (self.beta_[:, 1] - self.beta_[:, 0]) / (self.beta_[:, 2] - self.beta_[:, 0]),
                0.5
            )
        })
        return df

    def predict_uncertainty(self, X: np.ndarray) -> pd.DataFrame:
        """
        Returns per-transaction prediction intervals [p_left, p_center, p_right],
        expected value, and uncertainty interval width.
        """
        P_tfn = self.predict_fuzzy(X)
        ev = dubois_prade_expected_value(P_tfn)
        return pd.DataFrame({
            "p_left": P_tfn[:, 0],
            "p_center": P_tfn[:, 1],
            "p_right": P_tfn[:, 2],
            "expected_value": ev,
            "uncertainty_width": P_tfn[:, 2] - P_tfn[:, 0]
        })

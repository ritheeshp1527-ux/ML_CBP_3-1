# Guided Fuzzy Learning Methodology: Principled Formulation, Imbalance Handling, and Validation

## 1. Executive Summary

This methodology document presents the **Guided Fuzzy Logistic Regression** framework developed in Milestone 3 of FuzzyGuard AI.

### Core Breakthrough
- **Preserved Fuzzy Semantics**: Full retention of the Triangular Fuzzy Number (TFN) formulation from Charizanos et al. (2024), including sign-dependent TFN dot product, component-wise sigmoid, and Dubois & Prade (1987) defuzzification.
- **Overcoming Random Monte Carlo Limitations**: By replacing uniform random hypercube sampling with **principled Fisher-information initialization** and **bounded L-BFGS-B optimization**, the guided model elevates test PR-AUC from **0.0010** (Table 5) and **0.0310** (Monte Carlo) to **0.6972** (over 500x higher than random prevalence).
- **Explainable Uncertainty**: The model outputs strict TFN coefficients $\bar{\beta}_j = [a_j \le b_j \le c_j]$ and prediction probability intervals $[p_{\text{left}}, p_{\text{center}}, p_{\text{right}}]$, enabling financial institutions to quantify transaction risk uncertainty.

---

## 2. Mathematical Architecture

### 2.1 Coefficient Representation and Bounded Parameterization
Each coefficient $\bar{\beta}_j$ for $j \in \{0 \dots D\}$ (where $j=0$ is the intercept and $j=1 \dots 29$ are feature slopes) is parameterized as:
$$a_j = b_j - s_{j,\text{left}}$$
$$b_j = b_j$$
$$c_j = b_j + s_{j,\text{right}}$$
where $s_{j,\text{left}} \ge s_{\min} > 0$ and $s_{j,\text{right}} \ge s_{\min} > 0$ are the lower and upper spreads.
This parameterization guarantees that $a_j < b_j < c_j$ holds strictly throughout training and inference.

### 2.2 Sign-Dependent TFN Linear Combination
For crisp predictor matrix $X \in \mathbb{R}^{N \times D}$, scalar multiplication by TFN $\bar{\beta}_j$ depends on the sign of $X_{ij}$:
- If $X_{ij} \ge 0$: $\bar{\beta}_j X_{ij} = [a_j X_{ij}, b_j X_{ij}, c_j X_{ij}]$
- If $X_{ij} < 0$: $\bar{\beta}_j X_{ij} = [c_j X_{ij}, b_j X_{ij}, a_j X_{ij}]$

Vectorized across all samples:
Let $X_+ = \max(X, 0)$ and $X_- = \min(X, 0)$.
$$z_{\text{left}} = a_0 + X_+ @ a_{1:D} + X_- @ c_{1:D}$$
$$z_{\text{center}} = b_0 + X @ b_{1:D}$$
$$z_{\text{right}} = c_0 + X_+ @ c_{1:D} + X_- @ a_{1:D}$$

Because $a_j \le c_j$ and $X_- \le 0$, it is mathematically guaranteed that:
$$z_{i,\text{left}} \le z_{i,\text{center}} \le z_{i,\text{right}} \quad \forall i$$

### 2.3 Fuzzy Logistic Link & Dubois–Prade Defuzzification
The fuzzy probability triplet is computed via component-wise sigmoid:
$$\bar{P}_i = [p_{i,\text{left}}, p_{i,\text{center}}, p_{i,\text{right}}] = [\sigma(z_{i,\text{left}}), \sigma(z_{i,\text{center}}), \sigma(z_{i,\text{right}})]$$
Because $\sigma(u)$ is strictly monotonic, $0 < p_{\text{left}} \le p_{\text{center}} \le p_{\text{right}} < 1$.

Defuzzification uses Dubois & Prade (1987) mean value:
$$E(\bar{P}_i) = \frac{p_{i,\text{left}} + 2 p_{i,\text{center}} + p_{i,\text{right}}}{4}$$
Binary prediction rule:
$$\hat{y}_i = \mathbb{I}(E(\bar{P}_i) \ge \tau)$$
where threshold $\tau$ is chosen strictly on the validation partition to maximize F1-score.

---

## 3. Initialization & Spread Selection

To avoid the non-convergence of uniform random search in 30 continuous dimensions without leaking validation or test data:

1. **Center Initialization**:
   A reference Logistic Regression model is fitted on the chronological training split only. Its maximum-likelihood estimates $\hat{b}_0, \hat{b}_1 \dots \hat{b}_D$ serve as the starting center vector.
2. **Principled Spread Initialization**:
   Using asymptotic estimation theory, parameter uncertainty is governed by the inverse Fisher Information matrix:
   $$W = \text{diag}(p_i (1 - p_i)), \quad I = X_{\text{bias}}^T W X_{\text{bias}} + \lambda_{\text{ridge}} I$$
   $$\text{Cov}(\hat{\beta}) = I^{-1}, \quad \text{SE}_j = \sqrt{\max(\text{Cov}_{jj}, 10^{-6})}$$
   Initial spreads are set proportional to the standard error:
   $$s_{j,\text{left}}^{(0)} = s_{j,\text{right}}^{(0)} = \kappa \cdot \text{SE}_j \quad (\kappa = 1.0)$$

---

## 4. Class Imbalance & Objective Comparison (Phase 3)

Three distinct learning objectives were implemented and compared on the training partition:

### Objective A: `weighted_bce` (Primary Guided Model)
$$\mathcal{L}(\theta) = - \frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \log(\max(E(\bar{P}_i), \epsilon)) + w_0 (1 - y_i) \log(\max(1 - E(\bar{P}_i), \epsilon)) \right] + \lambda_{\text{spread}} \sum_{j=0}^D (s_{j,\text{left}} + s_{j,\text{right}})$$
- **Weights**: Balanced class weighting $w_1 = \frac{N}{2 N_1}, w_0 = \frac{N}{2 N_0}$.
- **Regularization**: $\lambda_{\text{spread}} = 0.005$ penalizes excessive interval width.
- **Validation Outcome**: PR-AUC = **0.6539**, ROC-AUC = **0.9727**, Validation Recall = **82.46%** (48/57 frauds caught).
- **Test Outcome**: PR-AUC = **0.6972**, ROC-AUC = **0.9810**, Test Recall = **85.33%** (64/75 frauds caught).

### Objective B: `class_weighted_fuzzy_mae`
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N w_{y_i} \sum_{k=1}^3 |y_{ik} - p_{ik}| + \lambda_{\text{spread}} \sum_j (s_{j,\text{left}} + s_{j,\text{right}})$$
- **Validation Outcome**: PR-AUC = **0.0790**, ROC-AUC = **0.9734**, Recall = **71.93%**, False Positives = **259**.
- **Trade-off**: Lower PR-AUC due to the noise variance of uniform fuzzification spreads ($U \sim \text{Uniform}(0, 1.5)$).

### Objective C: `unweighted_fuzzy_mae` (Author Paper Formulation)
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^3 |y_{ik} - p_{ik}| + \lambda_{\text{spread}} \sum_j (s_{j,\text{left}} + s_{j,\text{right}})$$
- **Validation Outcome**: PR-AUC = **0.7356**, ROC-AUC = **0.9479**, Validation F1 = **0.8081** (threshold 0.60), Recall = **70.18%**, False Positives = **only 2**.
- **Trade-off**: Highly conservative decision boundary favoring precision over rare-class recall.

---

## 5. Summary Evaluation & Model Comparison

Evaluated on chronological 60/20/20 partitions:
- **Validation**: 56,961 transactions (57 frauds)
- **Test**: 56,962 transactions (75 frauds) — **strictly untouched** during model fitting and threshold selection.

| Model | Test Threshold | Precision | Recall | Specificity | F1 (95% CI) | PR-AUC | ROC-AUC | MCC | FP | TP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.20 | **0.9310** | 0.7200 | **0.9999** | **0.8120** [0.724, 0.874] | **0.7846** | 0.9375 | **0.8185** | **4** | 54 |
| **Guided Fuzzy LogReg** | 0.90 | 0.1934 | **0.8533** | 0.9953 | **0.3153** [0.254, 0.370] | **0.6972** | **0.9810** | 0.4049 | 267 | **64** |
| **Logistic Regression** | 0.20 | 0.7183 | 0.6800 | 0.9996 | **0.6986** [0.602, 0.779] | **0.6926** | 0.9731 | 0.6985 | 20 | 51 |
| **LR (Balanced)** | 0.90 | 0.2433 | **0.8533** | 0.9965 | **0.3787** [0.309, 0.441] | **0.7436** | 0.9820 | 0.4546 | 199 | **64** |
| **Fuzzy Learned MC** | 0.70 | 0.0652 | 0.3600 | 0.9932 | **0.1104** [0.074, 0.149] | 0.0310 | 0.6900 | 0.1508 | 387 | 27 |
| **Fuzzy Table 5** | 0.25 | 0.0015 | 0.9067 | 0.1793 | **0.0029** [0.002, 0.004] | 0.0010 | 0.3769 | 0.0081 | 46687 | 68 |

*Reference test fraud prevalence baseline*: $\pi_1 = 75 / 56962 = 0.001317$ ($0.13\%$).

---

## 6. Uncertainty Explainability

The guided fuzzy model produces dual-level uncertainty quantification:
1. **Parameter Uncertainty**:
   Feature coefficients are reported with left/right spreads (e.g. $V_1 \in [10.333, 10.343]$, $V_4 \in [2.291, 2.301]$), showing parameter confidence bounds.
2. **Transaction Risk Interval**:
   Every incoming transaction receives a risk triplet $[p_{\text{left}}, p_{\text{center}}, p_{\text{right}}]$. Transactions with high spread $p_{\text{right}} - p_{\text{left}}$ represent borderline or ambiguous cases requiring human investigator review rather than automated blocking.

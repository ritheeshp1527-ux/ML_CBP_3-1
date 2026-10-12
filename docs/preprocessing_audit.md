# Preprocessing Audit: Evidence, Transformations, and Reproducibility

This audit investigates the exact data preparation and preprocessing procedures used by Charizanos et al. (2024) in generating and evaluating the fuzzy logistic regression model and its published Table 5 coefficients.

---

## 1. Preprocessing Audit Matrix

| Feature or Operation | Evidence in Paper or Source Code | Exact Transformation | Status | Consequences for Reproducing Table 5 Inference |
| :--- | :--- | :--- | :--- | :--- |
| **Predictor Count & Selection** | Paper Section 2.1: *"The dataset contains 29 predictors, including the amount of each transaction, named V29."*<br>Table 5: Exactly 30 rows ($\bar{\beta}_0$ intercept + $\bar{\beta}_1 \dots \bar{\beta}_{29}$). | Exclude `Time` and `Class`. Use 29 predictors: $V_1$ through $V_{28}$, plus $V_{29} = \text{Amount}$. | **Confirmed** | Using 29 features matches Table 5 coefficient matrix dimensions ($30 \times 3$). |
| **Exclusion of `Time`** | Paper Section 2.1 & 2.4.2: `Time` (or `Timestamps`) is used exclusively for stratum separation in SRS ($t \le 50000$s vs $t > 50000$s). No $\bar{\beta}_{\text{Time}}$ exists in Table 5. | Dropped from regression feature matrix $X$. | **Confirmed** | Including `Time` in $X$ would cause dimensional mismatch with Table 5 coefficients. |
| **Feature Ordering** | Table 5 lists $\bar{\beta}_0$ through $\bar{\beta}_{29}$. Paper Section 3.2 explicitly discusses $V_{25}$ (odds ratio 2.17 for legitimate, matching $\bar{\beta}_{25} = -0.820$) and $V_{26}$ (odds ratio 1.82 for fraud, matching $\bar{\beta}_{26} = +0.581$). | Index 0: Intercept $\bar{\beta}_0$.<br>Indices 1–28: $V_1 \dots V_{28}$.<br>Index 29: $V_{29} = \text{Amount}$. | **Confirmed** | Explicit mapping verified: feature columns must strictly follow $[V_1 \dots V_{28}, \text{Amount}]$. |
| **Intercept Handling** | Table 5 gives $\bar{\beta}_0 = (-0.920, -0.587, 0.552)$. `online-fraud-detection` line 83: `b0 <- coef_matrix[1, , drop = FALSE]`. | Added as TFN bias to linear combination: $z = \bar{\beta}_0 + \sum \bar{\beta}_j X_j$. | **Confirmed** | Correctly incorporated as row 0 in coefficient matrix. |
| **Transformations on $V_1$–$V_{28}$** | Paper Section 2.1: $V_1$–$V_{28}$ are PCA components from European cardholder transactions. `online-fraud-detection` lines 181–182: `Xmtx <- training_data_df[, !(names(training_data_df) %in% response_var)]` (passes continuous features directly). | None. Continuous values used as-is. | **Confirmed** | Features are continuous PCA components (mean $\approx 0$, variance $\approx 1$ to $5$). |
| **Scaling / Transformation of `Amount` ($V_{29}$)** | Paper Section 2.1 mentions dollar values ($0.01 to $25,692, average $88). No normalization, standardization, or log transform is mentioned in paper text or author R code. | Unscaled raw dollar value in code; no explicit transform found in author repo. | **Unresolved** (Evidence favors raw unscaled, but scaling was tested in diagnostics) | In Table 5, $\bar{\beta}_{29} = (-0.157, -0.066, 0.009)$. With unscaled Amount ($\le 25000$), $-0.066 \times \text{Amount}$ causes logit collapse to $-500$, driving probabilities to 0. When standardized, collapse is reduced, but ROC-AUC remains $< 0.43$. |
| **Response Fuzzification** | Paper Section 2.3.1 & 3.2: $m = 2.5, l = 1.0, r = 1.0, I_L = 0.0, I_U = 1.5$. Code: `fuzzify_Y` with hardcoded `set.seed(123)`. | $Y=0 \implies [0 - 2.5 U, 0, 0 + 2.5 U]$<br>$Y=1 \implies [1 - 2.5 U, 1, 1 + 2.5 U]$ where $U \sim \text{Uniform}(0, 1.5)$. | **Confirmed** | Mathematical formulation confirmed; used during Monte Carlo loss evaluation. |
| **Sampling & Training Data Construction** | Paper Section 2.4.2 describes stratified random sampling by time threshold. Code (`SRS-handle-non-stationary-changes` lines 139–144) blends rows via convex combination: $W_t s_1 + (1-W_t) s_2$, then synthetically assigns Class labels (first 10 as 1, rest as 0). | Pairwise linear blending of feature rows between two sampled sets, followed by artificial label assignment. | **Confirmed in code**, **Discrepant with paper text** | Table 5 coefficients were optimized on a synthetically blended 5,000-row sample, disrupting natural feature-label correlations. |
| **Sufficiency of Table 5 for Reproduction** | Forward inference using Table 5 evaluated on validation set (56,961 transactions) and test set (56,962 transactions). | Evaluated under Raw, StandardScaler, MinMaxScaler, RobustScaler, and Log1p Amount. | **Confirmed Insufficient** | Table 5 coefficients produce inverted rankings ($ROC\text{-}AUC \approx 0.24 - 0.34$, $PR\text{-}AUC \approx 0.0008$, Recall $= 0.00$) because key fraud predictor signs in Table 5 are inverted (e.g. $\bar{\beta}_{V4} < 0$). |

---

## 2. In-Depth Root-Cause Analysis of Table 5 Performance

### 2.1 Coefficient Sign Inversion on Primary Fraud Indicators
Comparing the author's Table 5 center coefficients against standard logistic regression coefficients fitted on the chronological training set reveals critical discrepancies:
- **$V_4$**: Conventional LR yields $\beta_{V4} = +0.7042$ (or $+0.8566$ standardized), reflecting that fraudulent transactions exhibit significantly higher positive $V_4$. Table 5 center is **$-0.5180$** (opposite sign).
- **$V_{13}$**: Conventional LR yields $\beta_{V13} = -0.3422$. Table 5 center is **$+0.7330$** (opposite sign).
- **$V_{22}$**: Conventional LR yields $\beta_{V22} = +0.7103$. Table 5 center is **$-0.5340$** (opposite sign).
- **$V_{25}$**: Conventional LR yields $\beta_{V25} = -0.0160$. Table 5 center is **$-0.8200$** (massive negative).

Because the signs of dominant discriminant features are inverted, transactions with fraudulent characteristics receive lower logits than legitimate transactions, resulting in an inverted ROC-AUC ($0.3428$ on validation data).

### 2.2 Dimensionality and Interval Spread Dynamics
In 29 dimensions, each TFN linear combination accumulates vertex spreads:
$$z_{\text{left}} = a_0 + \sum_{X_j \ge 0} a_j X_j + \sum_{X_j < 0} c_j X_j$$
$$z_{\text{right}} = c_0 + \sum_{X_j \ge 0} c_j X_j + \sum_{X_j < 0} a_j X_j$$
Because $a_j < 0$ and $c_j > 0$ for almost all 29 features:
- Every term in $z_{\text{left}}$ is non-positive, accumulating large negative sums ($z_{\text{left}} \in [-30, -5]$), causing $p_{\text{left}} = \sigma(z_{\text{left}}) \approx 0.0$.
- Every term in $z_{\text{right}}$ is non-negative, accumulating large positive sums ($z_{\text{right}} \in [5, 30]$), causing $p_{\text{right}} = \sigma(z_{\text{right}}) \approx 1.0$.
- Consequently, for almost all transactions:
  $$E(\bar{P}) \approx \frac{0 + 2 p_{\text{center}} + 1}{4} = 0.25 + 0.5 p_{\text{center}}$$
The expected value is compressed into $[0.25, 0.75]$, and its rank ordering is identical to the rank ordering of $p_{\text{center}}$.

### 2.3 Impact of Monte Carlo Optimization on Imbalanced Data
The author's training code (`Optimize_Y_gen_cs2`) samples candidate coefficients uniformly at random in $[-1, 1]^{30}$ and selects the candidate minimizing MAE against fuzzified binary labels:
- In 30 dimensions, the volume of $[-1, 1]^{30}$ is $2^{30} \approx 1.07 \times 10^9$. Drawing 500 or 10,000 random points samples an infinitesimal fraction ($< 0.001\%$) of the parameter space.
- Because 99.8% of transactions are legitimate ($Y=0$, target center 0), candidates that suppress overall probabilities obtain very low MAE on the 99.8% majority class regardless of fraud discrimination.

---

## 3. Preprocessing Recommendations for Python Reproduction

1. **Feature Input**: Preserve the 29-feature schema $[V_1 \dots V_{28}, \text{Amount}]$ and exclude `Time`.
2. **Scaling Sensitivity**: When fitting fuzzy coefficients, test both unscaled and standardized continuous features. In both cases, fit scalers exclusively on the chronological training split.
3. **Partition Guardrails**: Never apply row-wise feature blending across strata to validation or test data. Evaluate models on the genuine chronological partitions.

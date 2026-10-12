# Guided Fuzzy Learning Audit & Mismatch Analysis

This document records the Phase 1 audit of the FuzzyGuard AI system before implementing the guided fuzzy-coefficient learning method. Every finding is labeled as **Confirmed**, **Inferred**, or **Unresolved**.

---

## 1. Audit Matrix of Existing Implementation and Methodology

| Aspect | Finding & Evidence | Status | Impact / Next Action |
| :--- | :--- | :--- | :--- |
| **Mathematical Formulation of Forward Inference** | `src/fuzzy_model.py` implements: `z_left = b0[0] + X_pos @ a + X_neg @ c`, `z_center = b0[1] + X @ b`, `z_right = b0[2] + X_pos @ c + X_neg @ a`, followed by component-wise sigmoid and Dubois–Prade EV $E(\bar{P}) = (p_1 + 2p_2 + p_3) / 4$. Matches author's R implementation with difference $< 1.11 \times 10^{-16}$. | **Confirmed** | Formulation is mathematically verified; retain as core TFN inference engine. |
| **Reproducibility of Baseline & Diagnostic Artifacts** | Recomputed metrics from `artifacts/metrics/*_scores.csv` match `baseline_results.json` and `fuzzy_training_results.json` to the last decimal (e.g. LR val recall = 0.4737, PR-AUC = 0.6504, ROC-AUC = 0.9678). | **Confirmed** | Previous pipeline outputs are fully reproducible and verified. |
| **Chronological Partition Guardrails** | Split is 60% train ($N=170,884$), 20% val ($N=56,961$), 20% test ($N=56,962$) by original `Time` row order. Test partition has 75 frauds and has never been used for training, feature selection, or threshold tuning. | **Confirmed** | Strict guardrail holds: test partition remains untouched until final frozen comparison. |
| **Feature Selection and Dimensionality** | 29 continuous predictors: $V_1 \dots V_{28}$ and $V_{29} = \text{Amount}$. `Time` is excluded from the regression feature matrix. Table 5 has 30 rows (intercept $\bar{\beta}_0$ + 29 slopes). | **Confirmed** | Retain 29 predictors $[V_1 \dots V_{28}, \text{Amount}]$. |
| **Monte Carlo Training Objective** | In author script `Optimize_Y_gen_cs2`, loss is unweighted MAE/MSE across 3 vertices: $\frac{1}{N} \sum_{i} \sum_{k=1}^3 |y_{ik} - p_{ik}|$, where $y$ is fuzzified with $m=2.5, U \sim \text{Uniform}(0, 1.5)$. | **Confirmed** | Exact author loss confirmed, but suffers severe misalignment with fraud detection (see Section 2). |
| **Class Imbalance in Author Objective** | Author loss is completely unweighted. With 0.2% fraud prevalence, legitimate transactions account for 99.8% of the loss gradient/penalty. Missing 100% of frauds incurs negligible penalty. | **Confirmed** | Author objective fails under extreme class imbalance without weighting or ranking adaptation. |
| **Noise-to-Signal Ratio in Response Fuzzification** | Author fuzzifies $Y \in \{0, 1\}$ by adding uniform spreads $m \cdot U \in [0, 3.75]$ (mean 1.875). The artificial noise magnitude exceeds the true binary signal difference ($1.0$). | **Confirmed** | Minimizing MAE against random uniform spreads attempts to fit noise seeds rather than fraud patterns. |
| **Table 5 Inverted Discrimination on Genuine Data** | Table 5 center coefficients have reversed signs on decisive fraud predictors ($\beta_{V4} = -0.518$ vs $+0.704$ in empirical LR; $\beta_{V13} = +0.733$ vs $-0.342$; $\beta_{V22} = -0.534$ vs $+0.710$). Table 5 yields validation ROC-AUC $\approx 0.35 - 0.41$. | **Confirmed** | Published coefficients were fitted on an artificially blended sample and cannot discriminate fraud on genuine data. |
| **Author SRS Implementation Discrepancy** | Author R script `SRS-handle-non-stationary-changes` linearly blended feature vectors row-by-row ($W_t s_1 + (1-W_t) s_2$) and synthetically forced labels (10 ones, 4990 zeros). | **Confirmed in code**, **Discrepant with paper text** | Blending broke genuine transaction relationships. Do not apply blending to project splits. |

---

## 2. In-Depth Objective Function Mismatch Analysis

### 2.1 The Class Imbalance Blind Spot
The author's objective function in `Fuzzy Logistic Regression`:
$$\mathcal{L}_{\text{MAE}}(\bar{\beta}) = \frac{1}{N} \sum_{i=1}^N \left( |y_{i,\text{left}} - p_{i,\text{left}}| + |y_{i,\text{center}} - p_{i,\text{center}}| + |y_{i,\text{right}} - p_{i,\text{right}}| \right)$$
In our training partition, $N_{\text{fraud}} = 360$ out of $N = 170,884$ ($0.211\%$).
The loss decomposes into:
$$\mathcal{L}_{\text{MAE}} = 0.99789 \cdot \bar{\mathcal{L}}_{\text{legitimate}} + 0.00211 \cdot \bar{\mathcal{L}}_{\text{fraud}}$$
Because legitimate transactions comprise $99.79\%$ of the dataset, any candidate coefficient vector that drives $p \to 0$ achieves near-optimal loss on the majority class. A model that predicts zero probability for all transactions achieves $\bar{\mathcal{L}}_{\text{legitimate}} \approx 3.75$ and total loss $\approx 3.752$, while missing every single fraud transaction. The objective provides negligible gradient or Monte Carlo reward for identifying the 360 frauds.

### 2.2 Artificial Noise vs Discriminative Signal
In Charizanos et al. (2024), fuzzification maps binary $Y \in \{0, 1\}$ into:
- $Y=0 \implies [-2.5 U_l, 0, +2.5 U_r]$ with $U \sim \text{Uniform}(0, 1.5)$
- $Y=1 \implies [1 - 2.5 U_l, 1, 1 + 2.5 U_r]$ with $U \sim \text{Uniform}(0, 1.5)$

The expected vertex width is $2 \times 2.5 \times 0.75 = 3.75$. The target values for legitimate transactions have $y_{\text{right}} \in [0, 3.75]$, and for fraudulent transactions $y_{\text{left}} \in [-2.75, 1]$.
Consequently:
1. $y_{\text{right}}$ for legitimate transactions is frequently larger than $y_{\text{center}}$ for fraudulent transactions ($1.0$).
2. The model spends its parameter capacity trying to fit random uniform draws $U$ rather than separating fraud from non-fraud.

### 2.3 Uniform Monte Carlo in 30-Dimensional Continuous Space
The author's learning loop generates random uniform coefficients in $[-1, 1]^{30}$.
- The volume of $[-1, 1]^{30}$ is $2^{30} \approx 1.07 \times 10^9$.
- Drawing 5,000 random points samples a fraction of $5 \times 10^{-6}$ of the parameter space.
- The probability of randomly drawing the correct signs across 30 dimensions is $(1/2)^{30} \approx 9.3 \times 10^{-10}$.
- This explains why pure Monte Carlo search without guidance cannot learn optimal decision boundaries in 30 continuous dimensions.

---

## 3. Principles for Guided Fuzzy Learning (Phase 2 & Phase 3)

To resolve the identified mismatches while preserving the triangular fuzzy number framework:

1. **Principle 1 — Discriminative Initialization**:
   Fit conventional Logistic Regression on the training partition only. Its maximum-likelihood estimates $\hat{\beta}_{\text{LR}}$ capture the true empirical log-odds slopes. Use $\hat{\beta}_{\text{LR}}$ as the reference center for fuzzy coefficient learning.
2. **Principle 2 — Valid Triangular Fuzzy Number Spreads**:
   Define fuzzy coefficient triplets $\bar{\beta}_j = (b_j - s_{j,\text{left}}, b_j, b_j + s_{j,\text{right}})$ where $s_{j,\text{left}}, s_{j,\text{right}} \ge 0$, guaranteeing $a_j \le b_j \le c_j$.
   Parameterize spreads $s_j$ via standard errors from the Fisher information matrix or bounded optimization on training data.
3. **Principle 3 — Class-Weighted & Probability-Aligned Objective**:
   To counteract the 99.8% imbalance, optimize fuzzy coefficients using a weighted objective:
   $$\mathcal{L}_{\text{guided}} = - \frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \log(E(\bar{P}_i)) + w_0 (1 - y_i) \log(1 - E(\bar{P}_i)) \right] + \lambda \sum_{j} (s_{j,\text{left}} + s_{j,\text{right}})$$
   where $w_1 = \frac{N}{2 N_1}, w_0 = \frac{N}{2 N_0}$ (balanced class weighting), or compare with weighted fuzzy MAE.
4. **Principle 4 — Uncertainty Preservation**:
   The resulting model must output the full TFN triplet $[p_{\text{left}}, p_{\text{center}}, p_{\text{right}}]$ and expected value $E(\bar{P})$, providing both high discriminative accuracy and explicit uncertainty bounds.

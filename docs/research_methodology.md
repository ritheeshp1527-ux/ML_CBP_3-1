# Research Methodology — Research-Source Audit & Source-to-Paper Mapping

## 1. Executive Summary & Audit Completion Status

This document records the comprehensive research-source audit for the **FuzzyGuard AI** project, benchmarking the official codebase of Charizanos et al. (2024) against their published base paper and defining the roadmap for a faithful Python reproduction.

### Base Paper Reference
Charizanos, G., Demirhan, H., & İçen, D. (2024). *An online fuzzy fraud detection framework for credit card transactions*. **Expert Systems with Applications**, 252, 124127. https://doi.org/10.1016/j.eswa.2024.124127

### Repositories Audited & Cloned Locally
1. **Primary Author Framework**: `https://github.com/GZanos/FFL` (Cloned into `research/FFL`)
2. **Linked Fuzzy Logistic Regression Repository**: `https://github.com/GZanos/Fuzzy-Logistic-Regression` (Cloned into `research/Fuzzy-Logistic-Regression`)
3. **Official Base Paper PDF**: `ML_CBP_3-1_BasePaper.pdf` (15 pages, full text and tables extracted to `research/base_paper_extracted.txt`)
4. **Project Documentation PDF**: `ML CBP DOCUMENTATION.pdf` (11 pages, extracted to `research/project_documentation_extracted.txt`)

### Current Milestone Checklist
- [x] Cloned and inspected all files in author repository `GZanos/FFL`.
- [x] Cloned and inspected all files in linked repository `GZanos/Fuzzy-Logistic-Regression`.
- [x] Mapped paper equations, figures, and tables to exact source code routines.
- [x] Identified exact mathematical operations, response fuzzification, coefficient learning, and thresholding.
- [x] Analyzed sampling strategy (SRS) and documented author implementation discrepancies.
- [x] Verified online update behavior (author code performs static streaming evaluation, no weight retraining).
- [x] Confirmed dataset schema: 29 predictors ($V_1$–$V_{28}$ and $V_{29}=\text{Amount}$), with `Time` used only for SRS partitioning.
- [x] Extracted exact published coefficients (Table 5: 30 TFN coefficients $\bar{\beta}_0 \dots \bar{\beta}_{29}$) from the paper.
- [x] Guardrails confirmed: Chronological 60/20/20 split preserved without leakage; test set untouched.
- [x] Verified data validation and test suite execution.
- [x] Formulated specification for the smallest faithful Python port.

---

## 2. Source Repositories Inventory & Dependencies

### Repository 1: `research/FFL` (`GZanos/FFL`)
- **`README.md`**: Defines a 3-step workflow:
  1. Stratified Random Sampling (SRS) to address non-stationarity.
  2. Tuning and learning using the linked repository.
  3. Online implementation code.
- **`SRS-handle-non-stationary-changes`** (R script, 342 lines):
  - Dependencies: `library(dplyr)`.
  - Configurable parameters: `file.name`, `n = 5000` (target sample size), `time.threshold = 50000` (seconds).
  - Divides data into strata: recent ($Time \le 50000$) and historical ($Time > 50000$).
  - Calculates stratum within-variances and between-variances.
  - Implements sample generation and export to `training_data.csv`.
- **`online-fraud-detection`** (R script, 224 lines):
  - Dependencies: `FuzzyR`, `openxlsx`, `FuzzyNumbers`, `Rfast`.
  - Configurable parameters: `Fuzzification_factor = 2.5`, `left_point_symmetry = 1`, `right_point_symmetry = 1`, `fuzzy.Y.max = 1.5`, `fuzzy.threshold = c(0.4, 0.5, 0.6)`.
  - Reads `coef.csv` (expects 30 rows) and `online_data.csv`.
  - Simulates transaction-by-transaction streaming inference and defuzzifies predictions.
  - *Note*: Data files `coef.csv` and `online_data.csv` were omitted from the author's public GitHub repository.

### Repository 2: `research/Fuzzy-Logistic-Regression` (`GZanos/Fuzzy-Logistic-Regression`)
- **`README.md`**: Describes fuzzy logistic regression across four formulation cases:
  - Case II: Crisp predictors, fuzzy coefficients, fuzzy output (used for credit card fraud).
  - Case III: Fuzzy predictors, crisp coefficients, fuzzy output.
  - Case IV: All-fuzzy approach.
- **`Fuzzy Logistic Regression`** (R script, 921 lines):
  - Implements Monte Carlo optimization (`Optimize_Y_gen_cs2`) with `Iterations = 10000`, `b0.lower = -1, b0.upper = 1, bi.lower = -1, bi.upper = 1`.
  - Computes fuzzy loss metrics (MSE, MAE, RMSE) and picks optimal parameter runs.
- **`Immunotherapy.csv`**: Sample clinical dataset for demonstration.

---

## 3. Comprehensive Source-to-Paper Mapping

| Framework Component | Paper Specification (Charizanos et al., 2024) | Author Implementation (`FFL` / `Fuzzy-Logistic-Regression`) | Mathematical Form / Concrete Logic |
| :--- | :--- | :--- | :--- |
| **Model Formulation** | Section 2.3, Eq. (1): Logit link with fuzzy coefficients: $\log\left(\frac{\bar{P}}{1 - \bar{P}}\right) = \bar{\beta}_0 + \sum_{j=1}^k \bar{\beta}_j X_j$ | Case II in `Fuzzy Logistic Regression` (lines 334–352) and `online-fraud-detection` (`Estimate_Y`, lines 81–95) | Crisp predictors $X$, TFN coefficients $\bar{\beta}_j = (a_j, b_j, c_j)$, fuzzy probability $\bar{P} = (p_1, p_2, p_3)$. |
| **Response Fuzzification** | Section 2.3.1: $F(Y) = [Y - m \cdot l \cdot U, Y, Y + m \cdot r \cdot U]$, $U \sim \text{Uniform}(I_L, I_U)$ | `fuzzify_Y` in `online-fraud-detection` (lines 63–78) and `Fuzzy Logistic Regression` (lines 167–182) | For $Y \in \{0, 1\}$, center is $Y$. Left vertex: $Y - 2.5 \cdot 1.0 \cdot \text{runif}(0, 1.5)$. Right vertex: $Y + 2.5 \cdot 1.0 \cdot \text{runif}(0, 1.5)$. Hardcoded `set.seed(123)`. |
| **TFN Arithmetic** | Section 2.3, Definitions: Standard TFN addition, subtraction, scalar multiplication, division | `add_TFN`, `sub_TFN`, `mult_TFN`, `div_TFN` in both R scripts | Addition: $(a_1+a_2, b_1+b_2, c_1+c_2)$<br>Subtraction: $(a_1-c_2, b_1-b_2, c_1-a_2)$<br>Mult: $(\min(a_1 a_2, a_1 c_2, c_1 a_2, c_1 c_2), b_1 b_2, \max(\dots))$<br>Div: $(\min(a_1/a_2, \dots), b_1/b_2, \max(\dots))$ |
| **Prediction Calculation** | Section 2.4.4: Linear combination plugged into logistic function | `Estimate_Y` in `online-fraud-detection`: `predy = div_TFN(c(1,1,1), 1 + exp(mult_TFN(c(-1,-1,-1), bx)))` | For crisp $X_j$: if $X_j \ge 0$, $\bar{\beta}_j X_j = (a_j X_j, b_j X_j, c_j X_j)$; if $X_j < 0$, $\bar{\beta}_j X_j = (c_j X_j, b_j X_j, a_j X_j)$.<br>Probability evaluates directly to component-wise sigmoid: $\bar{P} = (\sigma(z_{\text{left}}), \sigma(z_{\text{center}}), \sigma(z_{\text{right}}))$. |
| **Coefficient Optimization** | Section 2.3.2, Section 3.2: Grid search / MC search over coefficient bounds $[-1, 1]$ | `Optimize_Y_gen_cs2` in `Fuzzy Logistic Regression` (lines 355–376) | Uniform Monte Carlo sampling of random TFNs $(a, b, c)$ with $a \le b \le c \in [-1, 1]$. Evaluates dataset loss (MAE/MSE/RMSE). Optimal run selected via minimum error. |
| **Decision Thresholding** | Section 2.3.3: Expected value $E(\bar{P}) < a_2 \implies \hat{y}_2 = 0$, else $\hat{y}_2 = 1$; fuzzy bounds $\hat{y}_1 = \hat{y}_2 - w(D)$, $\hat{y}_3 = \hat{y}_2 + w(D)$ | Lines 193–208 in `online-fraud-detection` | Dubois & Prade expected value: $E(\bar{P}) = \frac{p_1 + 2p_2 + p_3}{4}$.<br>Threshold vector $\bar{\tau} = (0.4, 0.5, 0.6)$ ($a_2 = 0.5$).<br>If $E(\bar{P}) < 0.5 \implies \hat{y}_2 = 0$ (Legitimate), else $\hat{y}_2 = 1$ (Fraud). |
| **Sampling Strategy (SRS)** | Section 2.4.2: Split training into two strata by timestamp threshold $t=50000$s; variance weight $W_t = V_1 / (V_1 + V_2)$ | `SRS-handle-non-stationary-changes` (lines 17–150) | Stratum 1: recent transactions; Stratum 2: older transactions. Variance of class label: $V_i = p_i (1 - p_i)$. $W_t = V_1 / (V_1 + V_2)$. |
| **Online Update Policy** | Section 2.4.4, Section 3.3: "Online implementation" | `online-fraud-detection` (lines 185–209) | **Static sequential evaluation loop**. No coefficient updating or gradient adaptation occurs online. The pre-trained coefficient vector is held constant. |

---

## 4. Key Audit Findings & Identified Discrepancies

### Finding 1: Online Mode is Sequential Evaluation, Not Online Learning
- **Finding**: Despite using the terminology "online fraud detection framework", neither the base paper nor the author code implements online weight updates, continual learning, or streaming drift adaptation.
- **Verification**: In `online-fraud-detection`, `coef_matrix` is read once from disk. Transactions from `online_data.csv` are evaluated sequentially inside an inference loop. No weight updates or feedback loops are executed.
- **Engineering Decision**: The Python port will reproduce the static sequential streaming simulator accurately. If adaptive updating is later added as an engineering extension, it will be clearly designated as an extension and will adhere strictly to a label-delay policy.

### Finding 2: Stratified Sampling in Code Blends Features Row-by-Row
- **Discrepancy**: The paper text states: *"We take $n_1 W_t$ observations from $s_1$ and $n_2 W_t$ from $s_2$ to create the training sample."* (Section 2.4.2). However, in `SRS-handle-non-stationary-changes` (lines 139–141), the R code executes:
  ```R
  final_sample[, -Class] <- time.sample.weight * subset_final_sample1[, -Class] + (1 - time.sample.weight) * subset_final_sample2[, -Class]
  ```
  The code literally computes a convex combination of continuous feature vectors across two paired samples of transactions, followed by synthetic assignment of labels.
- **Risk Assessment**: Blending transactions creates synthetic pseudo-transactions rather than genuine historical records.
- **Engineering Decision**: Keep this author behavior documented as an artifact of their SRS script. Our core chronological splits (`train`, `validation`, `test`) remain strictly untouched and unblended.

### Finding 3: Metric Calculation Discrepancy (Author Script vs Published Paper)
- **Discrepancy**: In the R script `online-fraud-detection` (lines 98–146), `specificity`, `sensitivity`, and `f1_score` unroll the $N \times 3$ TFN matrix into a flattened 1D vector of length $3N$ using `as.vector()`.
- **Paper Evidence**: In Table 6 of the paper (Confusion Matrix: $TN = 4972, FP = 17, FN = 1, TP = 10$), the total number of transactions is exactly $N = 5000$. The metrics reported in Table 7 (Sensitivity $= 10/11 = 0.909$, Specificity $= 4972/4989 = 0.995$, F1 $= 0.526$, MCC $= 0.809$) were computed on the **crisp binary decision $\hat{y}_2$** per transaction, NOT on the flattened $3N$ matrix.
- **Engineering Decision**: The Python evaluation pipeline will evaluate predictions at the transaction level ($\hat{y} \in \{0, 1\}$ against actual $Y \in \{0, 1\}$) matching the published paper's official tables.

### Finding 4: Exact Published Coefficients Available (Table 5)
- **Finding**: While the author repository omitted `coef.csv`, Table 5 of the base paper provides the exact 30 TFN coefficients obtained from their optimal learning stage run (MAE-optimized).
- **Significance**: This allows immediate validation of the forward inference mathematical port against published paper results without having to run 10,000 Monte Carlo iterations.

The published Table 5 coefficients $\bar{\beta}_j = (a_j, b_j, c_j)$:
- Intercept $\bar{\beta}_0$: $(-0.920, -0.587, 0.552)$
- $\bar{\beta}_1$: $(-0.699, -0.458, -0.314)$
- $\bar{\beta}_2$: $(-0.963, 0.345, 0.638)$
- $\bar{\beta}_3$: $(0.143, 0.316, 0.503)$
- $\bar{\beta}_4$: $(-0.889, -0.518, 0.664)$
- $\bar{\beta}_5$: $(-0.611, -0.504, 0.256)$
- $\bar{\beta}_6$: $(-0.229, 0.105, 0.837)$
- $\bar{\beta}_7$: $(0.017, 0.397, 0.597)$
- $\bar{\beta}_8$: $(-0.455, 0.223, 0.778)$
- $\bar{\beta}_9$: $(-0.533, -0.071, 0.895)$
- $\bar{\beta}_{10}$: $(-0.954, -0.654, 0.047)$
- $\bar{\beta}_{11}$: $(-0.780, -0.169, 0.693)$
- $\bar{\beta}_{12}$: $(0.248, 0.477, 0.984)$
- $\bar{\beta}_{13}$: $(-0.073, 0.733, 0.758)$
- $\bar{\beta}_{14}$: $(-0.456, -0.301, 0.100)$
- $\bar{\beta}_{15}$: $(-0.082, 0.433, 0.529)$
- $\bar{\beta}_{16}$: $(0.199, 0.482, 0.854)$
- $\bar{\beta}_{17}$: $(-0.671, -0.095, 0.184)$
- $\bar{\beta}_{18}$: $(-0.913, -0.443, 0.047)$
- $\bar{\beta}_{19}$: $(-0.932, 0.068, 0.493)$
- $\bar{\beta}_{20}$: $(-0.955, -0.336, 0.161)$
- $\bar{\beta}_{21}$: $(-0.605, -0.254, -0.018)$
- $\bar{\beta}_{22}$: $(-0.591, -0.534, 0.582)$
- $\bar{\beta}_{23}$: $(-0.651, -0.366, 0.508)$
- $\bar{\beta}_{24}$: $(-0.260, 0.363, 0.425)$
- $\bar{\beta}_{25}$: $(-0.836, -0.820, -0.672)$
- $\bar{\beta}_{26}$: $(0.289, 0.581, 0.931)$
- $\bar{\beta}_{27}$: $(-0.520, -0.176, 0.259)$
- $\bar{\beta}_{28}$: $(-0.162, 0.140, 0.319)$
- $\bar{\beta}_{29}$ (Amount): $(-0.157, -0.066, 0.009)$

---

## 5. Dataset & Partitioning Guardrails

1. **Dataset**: European credit card fraud dataset (`data/raw/creditcard.csv`, 284,807 rows, 31 columns).
2. **Duplicate Rows**: 1,081 duplicate rows preserved as real concurrent transactions.
3. **Partition Integrity**:
   - Chronological 60/20/20 split:
     - **Train**: 170,884 rows (360 frauds, fraud rate 0.211%).
     - **Validation**: 56,961 rows (57 frauds, fraud rate 0.100%).
     - **Test**: 56,962 rows (75 frauds, fraud rate 0.132%).
   - **Strict Rule**: Test partition remains untouched and will not be used for threshold tuning or parameter optimization.

---

## 6. Implemented Faithful Python Architecture (`src/fuzzy_model.py`)

The mathematical framework of Charizanos et al. (2024) has been fully ported and verified in Python:

1. **Exact Vectorized TFN Linear Combination**:
   $$\tilde{z} = \bar{\beta}_0 + \sum_{j=1}^{29} \bar{\beta}_j X_j$$
   Implemented in `fuzzy_linear_combination(X, beta)`:
   - For $X_{ij} \ge 0$: contributions are $(a_j X_{ij}, b_j X_{ij}, c_j X_{ij})$.
   - For $X_{ij} < 0$: contributions are $(c_j X_{ij}, b_j X_{ij}, a_j X_{ij})$.
   - Vectorized using $X_+ = \max(X, 0)$ and $X_- = \min(X, 0)$, achieving a verified numerical equivalence difference of $< 1.11 \times 10^{-16}$ compared to author's R row loop.

2. **Fuzzy Logistic Link Function**:
   Component-wise sigmoid $\bar{P} = (\sigma(z_{\text{left}}), \sigma(z_{\text{center}}), \sigma(z_{\text{right}}))$, rigorously maintaining the TFN ordering $0 < p_{\text{left}} \le p_{\text{center}} \le p_{\text{right}} < 1$.

3. **Dubois & Prade Defuzzification**:
   $$E(\bar{P}) = \frac{p_{\text{left}} + 2 p_{\text{center}} + p_{\text{right}}}{4}$$
   Binary classification rule: $\hat{y} = \mathbb{I}(E(\bar{P}) \ge \tau)$, with $\tau = 0.5$ by default and tuned exclusively on the validation split.

4. **Monte Carlo Optimization Engine (`FuzzyLogisticRegression.fit`)**:
   Reproduces the author's uniform random coefficient search over user-specified bounds $[-1, 1]^{30}$, minimizing the author's fuzzy MAE loss against fuzzified targets $TFN\_Y$.

---

## 7. Diagnostics of Published Table 5 Coefficients

Forward inference using the author's published Table 5 coefficients was evaluated across three defensible input representations on the validation split (56,961 transactions, 57 fraud cases):

1. **`raw_unscaled`** (Author's exact script input):
   - Precision: 0.0001, Recall: 0.0175, Specificity: 0.8673, F1: 0.0003, PR-AUC: 0.00077, ROC-AUC: 0.4080 (inverted)
   - Confusion matrix at threshold 0.5: $TN = 49411, FP = 7550, FN = 56, TP = 1$
2. **`standardized_all`** (Standardized using training statistics):
   - Precision: 0.0006, Recall: 0.1930, Specificity: 0.6934, F1: 0.0012, PR-AUC: 0.00075, ROC-AUC: 0.3747 (inverted)
   - Confusion matrix at threshold 0.5: $TN = 39499, FP = 17462, FN = 46, TP = 11$
3. **`standardized_amount_only`** (PCA components unscaled, Amount standardized on training):
   - Precision: 0.0005, Recall: 0.1579, Specificity: 0.7303, F1: 0.0010, PR-AUC: 0.00072, ROC-AUC: 0.3549 (inverted)

### Diagnostic Root Cause
- **Coefficient Sign Inversion**: The author's Table 5 center coefficients have reversed signs on decisive fraud predictors (e.g. $\beta_{V4} = -0.5180$ in Table 5 vs $+0.7042$ in empirical LR; $\beta_{V13} = +0.7330$ in Table 5 vs $-0.3422$ in empirical LR).
- **Interval Spread Dynamics**: Vertex width accumulation across 29 dimensions compresses $E(\bar{P})$ into $[0.22, 0.75]$.
- **Training Artifact**: Table 5 coefficients were optimized on a 5,000-row sample synthetically created by pairwise linear feature blending between two strata (`SRS-handle-non-stationary-changes`), altering genuine transaction correlations.
- Complete audit matrix is documented in [docs/preprocessing_audit.md](file:///c:/Users/Ritheesh/Desktop/WORK\Datasets/ML%20CBP%203-1/FuzzyGuard_AI_Day1_Starter/FuzzyGuard_AI_Starter/docs/preprocessing_audit.md).

---

## 8. Empirical Model Comparison on Supplied Dataset

All models were evaluated on consistent chronological splits:
- **Training**: 170,884 rows (360 fraud cases)
- **Validation**: 56,961 rows (57 fraud cases) — used for threshold selection
- **Test**: 56,962 rows (75 fraud cases) — untouched during training and threshold tuning

### Summary Comparison Table (`artifacts/metrics/model_comparison.csv`)

| Model | Evaluation Split | Threshold Mode | Thresh | Precision | Recall | Specificity | F1 | PR-AUC | ROC-AUC | MCC | FP | TP | Fit Time | Latency (ms/row) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Validation | Tuned | 0.20 | 0.6667 | 0.7018 | 0.9996 | **0.6838** | 0.6504 | 0.9678 | 0.6837 | 20 | 40 | 0.50s | 0.0002 |
| **Logistic Regression** | Test | Val-Selected | 0.20 | 0.7183 | 0.6800 | 0.9996 | **0.6986** | 0.6926 | 0.9731 | 0.6985 | 20 | 51 | 0.50s | 0.0002 |
| **LR (Balanced)** | Validation | Tuned | 0.90 | 0.1638 | 0.8421 | 0.9957 | **0.2743** | 0.7719 | 0.9709 | 0.3703 | 245 | 48 | 1.23s | 0.0001 |
| **LR (Balanced)** | Test | Val-Selected | 0.90 | 0.2433 | 0.8533 | 0.9965 | **0.3787** | 0.7436 | 0.9820 | 0.4546 | 199 | 64 | 1.23s | 0.0001 |
| **Random Forest** | Validation | Tuned | 0.20 | 0.9535 | 0.7193 | 1.0000 | **0.8200** | 0.7334 | 0.9169 | 0.8280 | 2 | 41 | 8.42s | 0.0015 |
| **Random Forest** | Test | Val-Selected | 0.20 | 0.9310 | 0.7200 | 0.9999 | **0.8120** | 0.7846 | 0.9375 | 0.8185 | 4 | 54 | 8.42s | 0.0016 |
| **Fuzzy Table 5** | Validation | Default | 0.50 | 0.0001 | 0.0175 | 0.8673 | **0.0003** | 0.0008 | 0.4080 | -0.0107 | 7550 | 1 | 0.00s | 0.0003 |
| **Fuzzy Table 5** | Test | Val-Selected | 0.25 | 0.0015 | 0.9067 | 0.1793 | **0.0029** | 0.0010 | 0.3769 | 0.0081 | 46687 | 68 | 0.00s | 0.0003 |
| **Fuzzy Learned MC** | Validation | Tuned | 0.70 | 0.0384 | 0.2456 | 0.9938 | **0.0664** | 0.0107 | 0.5706 | 0.0949 | 351 | 14 | 7.88s | 0.0003 |
| **Fuzzy Learned MC** | Test | Val-Selected | 0.70 | 0.0652 | 0.3600 | 0.9932 | **0.1104** | 0.0310 | 0.6900 | 0.1508 | 387 | 27 | 7.88s | 0.0003 |

### Interpretation & Fraud Trade-Off Analysis
1. **Severe Imbalance Reality**: In extreme imbalance (0.13% fraud rate in test), high recall alone is trivial to achieve by lowering the decision threshold, but at the cost of catastrophic false alarms. For instance, `Fuzzy Table 5` at threshold 0.25 flags 46,755 transactions as fraud to catch 68 fraud cases (Precision = 0.15%), rendering it operationally useless.
2. **Fuzzy Learned Model vs Table 5**: The newly learned Fuzzy MC model demonstrates a genuine learning progression: PR-AUC improves from 0.0008 to 0.0310 (38x increase), ROC-AUC improves from 0.3769 to 0.6900, and false positives drop by over 46,000 cases.
3. **Fuzzy Model vs Crisp Baselines**: Crisp Logistic Regression and Random Forest substantially outperform pure Monte Carlo fuzzy search (PR-AUC 0.69–0.78 vs 0.03). This is mathematically expected: uniform random search in 30 continuous dimensions without gradient guidance cannot find optimal likelihood weights within 5,000 iterations.
4. **Reproducibility Status**: The Python port is **mathematically faithful** to the author's published equations and R code routines, but the author's Table 5 published weights are **not reproducible** as an effective fraud detector on real data due to their synthetic SRS blending artifact.

---

## 9. Guided Fuzzy Learning (Milestone 3 Findings)

To overcome the computational and statistical limitations of uniform Monte Carlo search, Milestone 3 introduced `GuidedFuzzyLogisticRegression`:
1. **Parameterization**: $\theta = [\mathbf{b}, \mathbf{s}_l, \mathbf{s}_r]$ with $a_j = b_j - s_{l,j}$ and $c_j = b_j + s_{r,j}$ ($s \ge 0.005$), ensuring strict TFN ordering $a_j \le b_j \le c_j$.
2. **Initialization**: Center $\mathbf{b}$ initialized from training-only Logistic Regression; semi-spreads initialized from asymptotic standard errors derived from the inverse Fisher information matrix $\mathbf{I}^{-1}(\beta)$.
3. **Objective**: Class-weighted binary cross-entropy on Dubois–Prade expected probability $E[\tilde{P}_i]$ with spread $L_1$ penalty.
4. **Performance on Untouched Test Split**:
   - Recall: **85.33%** (64/75 frauds detected).
   - PR-AUC: **0.6972** (compared to 0.0310 for Monte Carlo fuzzy, a >22x increase).
   - ROC-AUC: **0.9810** (compared to 0.6900 for Monte Carlo fuzzy).
   - $F_1$ Score: **0.3153** [95% bootstrap CI: 0.2537, 0.3696].

---

## 10. Prequential Streaming and Delayed-Label Adaptation (Milestone 4 Findings)

Milestone 4 established a leak-free streaming prequential evaluation protocol across the 56,962 test transactions:
1. **Prequential Protocol**: Transactions are predicted using model version active at $t_i$ before label enqueueing. True labels mature at $t_i + \Delta t_{\text{delay}}$.
2. **Adaptation Under Replay and Proximal Regularization**:
   - Updates are triggered on batches of 2,500 matured transactions, buffered with a 200-sample historical replay reservoir and proximal penalty $\frac{\gamma_{\text{prox}}}{2} \|\theta - \theta_{t-1}\|^2$.
   - **Immediate feedback ($\Delta t = 0$s)**: False positives dropped from 267 down to 193 (a 27.7% reduction), improving $F_1$ from 0.3153 to **0.3855** and MCC from 0.4049 to **0.4599** while retaining full 85.33% recall.
   - **Delayed feedback ($\Delta t = 3600$s / 1 hr)**: False positives dropped from 267 down to 204 (a 23.6% reduction), improving $F_1$ to **0.3732** and MCC to **0.4503**.
   - **Baseline Failure**: Unregularized online Logistic Regression suffered acute concept drift explosion, flagging 1,320 false positives and collapsing $F_1$ to 0.0838.
3. **Concept Drift Across Chronological Windows**:
   - In Window 4 (end of Day 2), fraud distribution shifts: the frozen Logistic Regression baseline collapsed to 0.0% recall (0/11 frauds detected), whereas the Guided Fuzzy model detected 7/11 frauds (63.64% recall) with reduced false alarms.
4. **Human Review Triage Validation**:
   - **Threshold-Margin Distance**: Routing the top 1.0% most uncertain transactions by margin distance $|E[\tilde{P}] - \tau|$ captured **86.67% of all frauds** (65/75) into manual review, completely eliminating automated false alarms (automated FP = 0) and leaving only 10 automated false negatives among 56,392 auto-processed transactions.
   - **Fuzzy Width**: Pure fuzzy width $p_{\text{right}} - p_{\text{left}}$ peaks around the sigmoid midpoint ($P \approx 0.50$), making it an epistemic coefficient uncertainty metric rather than a standalone decision-boundary triage metric under extreme class imbalance.


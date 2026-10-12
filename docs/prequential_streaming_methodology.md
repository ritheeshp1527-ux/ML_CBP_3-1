# FuzzyGuard AI — Milestone 4 Methodology: Prequential Streaming Evaluation and Delayed-Label Adaptation

**Date**: 2026-10-12  
**Status**: Completed and Independently Verified  
**Author**: Antigravity AI  

---

## 1. Objective and Methodological Framework

Milestone 4 implements a rigorous prequential (Test-Then-Train) streaming evaluation protocol for the guided fuzzy credit-card fraud detection framework. The primary objectives are:
1. Formulate a realistic, leak-free streaming evaluation protocol with configurable label delays.
2. Evaluate whether incremental online adaptation improves fraud detection under chronological distribution shift.
3. Compare the frozen guided fuzzy model, adaptive guided fuzzy model (immediate labels), adaptive guided fuzzy model (delayed labels), and online conventional linear baselines.
4. Analyze temporal concept drift across sequential evaluation windows.
5. Scientifically test whether fuzzy probability bounds [$p_{\text{left}}, p_{\text{right}}$] and threshold margin distance enable decision-support triage for human investigators.

---

## 2. Prequential Streaming Protocol (Test-Then-Train)

### Chronological Data Partitions
- **Training partition** ($T \in [0, 120396]$ s, rows 0 to 170,883): 170,884 transactions, 360 fraud cases. Used exclusively for initial model estimation and constructing a compact replay reservoir.
- **Validation partition** ($T \in [120396, 145247]$ s, rows 170,884 to 227,844): 56,961 transactions, 57 fraud cases. Used exclusively for threshold selection ($\tau = 0.90$), delay policy design, and triage exploration.
- **Test partition** ($T \in [145248, 172792]$ s, rows 227,845 to 284,806): 56,962 transactions, 75 fraud cases. Strictly untouched during design; evaluated prequentially.

### Protocol Distinction:
- For the **frozen model**, the test partition remains an untouched static held-out evaluation set.
- For the **adaptive models**, the partition is a **prequential evaluation stream (Test-Then-Train)**. Each transaction is tested before its label is made available for subsequent training.

### Simulation Protocol Mechanics:
For each incoming transaction $i$ arriving at timestamp $t_i$:
1. **Prequential Prediction**: The transaction $\mathbf{x}_i$ is scored using the model parameters active at $t_i$ *prior* to label ingestion. The prediction record logs:
   $$\{t_i, \text{index}_i, p_{\text{left},i}, p_{\text{center},i}, p_{\text{right},i}, E[\tilde{P}_i], w_i, \hat{y}_i, y_i, \text{version}_t\}$$
2. **Label Enqueueing**: The transaction label $y_i$ is placed in a FIFO pending queue with maturity timestamp:
   $$t_{\text{matured}} = t_i + \Delta t_{\text{delay}}$$
3. **Maturity Check**: Transactions in the pending queue with $t_{\text{matured}} \le t_i$ are drained into the matured training buffer.
4. **Adaptive Update Trigger**: When the accumulated buffer of newly matured labeled transactions reaches $B = 2,500$ transactions, an incremental model update is executed:
   - To counteract catastrophic forgetting on extreme class imbalance ($0.13\%$ fraud), the update buffer is combined with a 200-sample balanced replay reservoir from the training partition.
   - The parameters $\theta = [\mathbf{b}, \mathbf{s}_l, \mathbf{s}_r]$ are updated via bounded `L-BFGS-B` under a proximal penalty:
     $$\mathcal{L}_{\text{online}}(\theta) = \mathcal{L}_{\text{BCE}}(\theta; \mathcal{B} \cup \mathcal{R}) + \lambda_{\text{spread}} \sum (s_{l,j} + s_{r,j}) + \frac{\gamma_{\text{prox}}}{2} \|\theta - \theta_{t-1}\|_2^2$$
   - `model_version` is incremented, and an audit record is logged.

---

## 3. Evaluated Adaptation Strategies (Chronological Test Stream)

Evaluation across all 56,962 test transactions (75 frauds, prevalence: $0.1317\%$):

| Strategy | Adaptation Mode | Label Delay | Model Updates | Recall | Precision | $F_1$ Score | PR-AUC | ROC-AUC | MCC | False Positives | False Negatives | Adaptation Time | Latency / tx |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Guided Fuzzy (Frozen)** | Frozen | N/A | 0 | **85.33%** | 19.34% | 0.3153 | **0.6972** | **0.9810** | 0.4049 | 267 | 11 | 0.00s | 0.037 ms |
| **Guided Fuzzy (Immediate)** | Online | 0s | 22 | **85.33%** | **24.90%** | **0.3855** | 0.6943 | 0.9725 | **0.4599** | **193** | 11 | 3.79s | 0.033 ms |
| **Guided Fuzzy (Delayed 1h)** | Online | 3,600s | 20 | **85.33%** | 23.88% | 0.3732 | 0.6958 | 0.9731 | 0.4503 | 204 | 11 | 3.72s | 0.035 ms |
| LogReg Baseline (Frozen) | Frozen | N/A | 0 | 25.33% | **95.00%** | 0.4000 | 0.5181 | 0.9770 | 0.4903 | **1** | 56 | 0.00s | 0.100 ms |
| LogReg Baseline (Adaptive) | Online | 0s | 22 | 81.33% | 4.42% | 0.0838 | 0.1568 | 0.9551 | 0.1863 | 1,320 | 14 | 0.15s | 0.100 ms |

### Key Experimental Discoveries:
1. **Online Adaptation Substantially Reduces False Alarms**:
   - Both adaptive guided fuzzy models maintain the exact same high fraud recall (**85.33%**, 64/75 frauds detected).
   - Adaptive updating reduces false positives from 267 down to 193 (immediate) and 204 (1-hour delay) — an immediate **27.7% reduction in false alerts**.
   - $F_1$ score increases from **0.3153** to **0.3855** (+0.0702), and MCC improves from **0.4049** to **0.4599**.
2. **Robustness to Delayed Labels**:
   - Under a realistic 1-hour delay ($\Delta t = 3,600$s), the guided fuzzy model retains almost the entire benefit of online adaptation ($F_1 = 0.3732$ vs $0.3855$), with false positives dropping to 204.
3. **Catastrophic Drift in Unregularized SGD Baseline**:
   - The conventional adaptive Logistic Regression baseline suffered severe alert explosion: false positives skyrocketed from 1 to 1,320, collapsing precision to 4.42% and $F_1$ to 0.0838.
   - In contrast, the guided fuzzy model's bounded spreads and proximal regularization guarantee numerical stability and monotonic TFN ordering.

---

## 4. Concept Drift Across Sequential Time Windows

The 7.65-hour test stream was partitioned into 4 chronological windows (~14,240 transactions each):

| Window | Time Span (hrs) | Frauds | Frozen Fuzzy Recall | Adaptive Fuzzy Recall | Frozen LR Recall | Adaptive LR Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W1** | 0.00 – 1.69 | 23 | **95.65%** (22/23) | **95.65%** (22/23) | 17.39% (4/23) | 91.30% (21/23) |
| **W2** | 1.69 – 3.44 | 30 | 83.33% (25/30) | 83.33% (25/30) | 40.00% (12/30) | 80.00% (24/30) |
| **W3** | 3.44 – 5.25 | 11 | **90.91%** (10/11) | **90.91%** (10/11) | 27.27% (3/11) | 90.91% (10/11) |
| **W4** | 5.25 – 7.65 | 11 | 63.64% (7/11) | 63.64% (7/11) | **0.00% (0/11)** | 54.55% (6/11) |

### Drift Findings:
- In Window 4 (the final period of Day 2), fraud distribution shifts: the Frozen Logistic Regression baseline **completely collapses to 0.0% recall** (0 out of 11 frauds detected).
- The Guided Fuzzy model retains detection capability (7/11 frauds detected, 63.64% recall) while cutting false alarms from 71 down to 49.

---

## 5. Validation of Fuzzy Uncertainty and Human Triage Policies

We investigated whether fuzzy uncertainty width $w_i = p_{\text{right},i} - p_{\text{left},i}$ or threshold margin distance $m_i = |E[\tilde{P}_i] - \tau|$ effectively routes transactions to manual investigation.

To avoid test-leakage, cutoffs were calibrated strictly on the validation stream (out-of-sample protocol) and compared against a fixed stream-rank budget simulation.

### Empirical Triage Results on Chronological Test Stream:

Base classifier errors before review: $\text{TP} = 64, \text{FP} = 267, \text{FN} = 11$.

| Protocol | Strategy | Target Budget | Actual Reviewed | Reviewed Frauds | Reviewed Prior TP | Reviewed Salvaged FN | Reviewed Cleared FP | Auto Misses (FN) | Auto Alarms (FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation-Calibrated** | Margin Uncertainty | 0.5% | 220 (0.39%) | 1 | 1 | 0 | 119 | 11 | 148 |
| **Validation-Calibrated** | Margin Uncertainty | 1.0% | 465 (0.82%) | 8 | 7 | 1 | 241 | 10 | 26 |
| **Validation-Calibrated** | Margin Uncertainty | 2.0% | 951 (1.67%) | 65 | 64 | 1 | 267 | 10 | **0** |
| **Validation-Calibrated** | Margin Uncertainty | 5.0% | 2,371 (4.16%) | 68 | 64 | 4 | 267 | 7 | **0** |
| Validation-Calibrated | Fuzzy Width | 1.0% | 406 (0.71%) | 1 | 0 | 1 | 3 | 10 | 264 |
| Validation-Calibrated | Fuzzy Width | 2.0% | 850 (1.49%) | 2 | 0 | 2 | 3 | 9 | 264 |
| Stream-Rank Cutoff | Margin Uncertainty | 1.0% | 570 (1.00%) | 65 | 64 | 1 | 267 | 10 | **0** |
| Stream-Rank Cutoff | Fuzzy Width | 1.0% | 570 (1.00%) | 1 | 0 | 1 | 3 | 10 | 264 |

### Critical Scientific Findings on Uncertainty & Triage:
1. **Triage is an Alert-Verification Queue, Not Fraud Discovery**:
   - Out of the 65 frauds reviewed under 2% validation-calibrated margin triage, **64 were frauds the model had ALREADY classified as positive ($P \ge 0.90$)**.
   - Human review salvaged only **1 previously missed false negative** at 1–2% budget, and **4 missed frauds** at 5% budget.
   - Therefore, triage does not uncover hidden fraud deep in the distribution; it routes the model's highest-scoring alerts to manual inspection.
2. **Oracle Human-Review Assumption**:
   - The reported elimination of automated false alarms (267 down to 0 at 2% budget) occurs because the classifier's false alarms are routed to the review queue and **assumed to be resolved with 100% human accuracy**.
   - This represents an **oracle-review simulation**, not empirical proof of human investigator accuracy in live operations.
3. **Fuzzy Width vs. Margin Distance**:
   - Fuzzy width $w_i = \sigma(z_{r,i}) - \sigma(z_{l,i})$ mathematically peaks when linear combinations $z_i \approx 0$ (probability $P \approx 0.50$).
   - Under extreme class imbalance with decision threshold $\tau = 0.90$, transactions with highest fuzzy width lie in the intermediate zone ($P \in [0.20, 0.60]$), not near $\tau = 0.90$.
   - **Conclusion**: Pure fuzzy width quantifies epistemic coefficient ambiguity across features, but does not serve as an effective standalone triage metric unless combined with boundary proximity.

---

## 6. Methodological Limitations

1. **Short Chronological Test Window**:
   - The test partition spans 7.65 hours (~27,500s). While realistic for micro-drift, macroeconomic macro-drift (weeks or months) cannot be observed within a 48-hour dataset.
2. **Simulated Label Arrival**:
   - In retail banking, chargebacks take 30 to 90 days to confirm. A 1-hour delay simulates fast automated dispute signals, not full chargeback cycles.
3. **Replay Buffer Dependency**:
   - Online adaptation on imbalanced streams requires replay memory to prevent catastrophic forgetting. Without replay memory, online updates degrade rapidly.

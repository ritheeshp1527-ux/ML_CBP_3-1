# FuzzyGuard AI — Online Fuzzy Credit Card Fraud Detection

An end-to-end research reproduction, guided fuzzy learning framework, prequential streaming adaptation simulator, and interactive research dashboard for credit card fraud detection under severe class imbalance and label delay.

Based on the research and methodology audit of:
> Charizanos, G., Demirhan, H., & İçen, D. (2024). *An online fuzzy fraud detection framework for credit card transactions*. Expert Systems with Applications, 252, 124127.

---

## 1. Project Overview

FuzzyGuard AI provides:
- **Data Pipeline**: Strict chronological 60/20/20 train/validation/test evaluation on 284,807 transactions (492 frauds, 0.173% prevalence) without lookahead leakage.
- **Published Paper Reproduction Audit**: Independent mathematical inspection of the 2024 authors' R implementation identifying the synthetic stratum convex-blending error that inverted feature coefficient signs.
- **Guided Fuzzy Learning**: Statistically grounded Fisher information initialization with bounded Triangular Fuzzy Numbers (TFN), improving PR-AUC from 0.0310 to 0.6972 (>22× baseline prevalence) and catching 64 of 75 test frauds.
- **Prequential Streaming Evaluation**: Realistic online test-then-train streaming protocol under a 1-hour ($\Delta t = 3600\text{s}$) dispute feedback delay with historical replay memory, cutting false alarms by 23.6% (267 down to 204).
- **Human-Review Triage Engine**: Workload evaluation comparing margin-based triage against fuzzy-width triage under finite review capacity budgets (0.5%–5.0%).
- **Interactive Research Dashboard**: Full-stack application (FastAPI backend + React/Vite/Tailwind frontend) featuring 7 specialized analytical views, Dubois–Prade attribution analysis, operational label masking, and custom transaction testing.

---

## 2. Repository Structure

```text
├── artifacts/              # Model weights, predictions (parquet), evaluation metrics (csv/json)
│   ├── metrics/            # Baseline, streaming, drift, and triage metrics
│   └── models/             # Trained guided fuzzy model, scaler, baselines
├── data/
│   └── raw/                # Place creditcard.csv here (excluded from git due to size)
├── docs/                   # Methodological audits, preprocessing analyses, prequential reports
├── frontend/               # React + Vite + Tailwind CSS + Recharts interactive dashboard
│   ├── src/                # Tab components, charts, and API client
│   └── package.json
├── research/               # Base paper extracts, R code audit notes, table scrapers
├── src/                    # Core Python pipeline
│   ├── api.py              # FastAPI backend server
│   ├── data.py             # Chronological split and validation
│   ├── guided_fuzzy_model.py # Bounded TFN logistic regression model
│   ├── streaming_adaptation.py # Prequential simulation with delayed queues
│   ├── train_baselines.py  # Logistic Regression & Random Forest baselines
│   └── evaluate_models.py  # Multi-metric offline and online evaluation
├── tests/                  # Pytest test suite (41 unit and regression tests)
└── requirements.txt        # Python backend dependencies
```

---

## 3. Quick Start & Execution

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Step 1: Python Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Run Tests
```bash
python -m pytest -v
```
*All 41 tests verify data splits, TFN ordering, prequential streaming order, label latency, and API endpoints.*

### Step 3: Launch Backend Server
```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
*Interactive API Swagger documentation is available at `http://127.0.0.1:8000/docs`.*

### Step 4: Launch Frontend Dashboard
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
*Access the research dashboard in your browser at `http://127.0.0.1:5173/`.*

---

## 4. Key Scientific Findings & Guardrails

1. **Synthetic Feature Blending in Author Code**:
   In the authors' R implementation (`SRS-handle-non-stationary-changes`, lines 139–144), stratified random sampling was applied with convex blending ($W_t s_1 + (1-W_t) s_2$) and overwritten class labels, causing catastrophic coefficient sign inversions on the genuine dataset ($\beta_{V4} = -0.518$ vs $+0.704$ empirical).
2. **Prequential Evaluation Protocol**:
   Every streamed transaction is evaluated strictly before model exposure. Delayed dispute labels arrive only after their maturity threshold ($\Delta t = 3600\text{s}$).
3. **Uncertainty Semantics**:
   Triangular fuzzy probability bounds $[p_{\text{left}}, p_{\text{right}}]$ represent epistemic ambiguity from parameter uncertainty via Dubois–Prade interval arithmetic; they are explicitly distinguished from statistically calibrated Bayesian or frequentist confidence intervals.
4. **Oracle Review Disclosure**:
   Human review triage results assume an oracle investigator (100% human accuracy on queued transactions) to isolate model triage efficiency from human error.
5. **Operational Label Masking**:
   The Transaction Explorer strictly masks true labels in operational viewing mode to replicate real-time inference conditions without hindsight bias.

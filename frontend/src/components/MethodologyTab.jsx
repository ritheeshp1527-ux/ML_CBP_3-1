import React from 'react';
import { 
  BookOpen, 
  AlertCircle, 
  CheckCircle2, 
  FileText, 
  Scale, 
  Clock, 
  ShieldAlert,
  GitBranch
} from 'lucide-react';

export default function MethodologyTab() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-cyan-400" />
          Research Methodology, Audit Findings & Scientific Limitations
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Detailed documentation of experimental protocols, discrepancies with the 2024 published paper, and deployment boundaries
        </p>
      </div>

      {/* Grid of Structured Note Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Card 1: Source Paper Replication & Audit Discrepancy */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md space-y-3">
          <div className="flex items-center gap-2 text-cyan-400 font-semibold text-sm">
            <FileText className="w-4 h-4" />
            <span>1. Published Paper Replication Audit</span>
          </div>
          <div className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <p>
              <strong>Paper Reference:</strong> Charizanos, G., Demirhan, H., &amp; Icen, D. (2024). <em>&ldquo;An Online Fuzzy Fraud Detection Framework for Credit Card Transactions.&rdquo;</em> Expert Systems with Applications, 252, 124127.
            </p>
            <p>
              <strong>Discrepancy Root Cause:</strong> In the author&apos;s R implementation (<code>SRS-handle-non-stationary-changes</code>), lines 139&ndash;144 performed a row-by-row convex feature combination across two strata (W<sub>t</sub> s<sub>1</sub> + (1 - W<sub>t</sub>) s<sub>2</sub>) and synthetically overwrote the <code>Class</code> column. This synthetic blending inverted coefficient signs on critical fraud features (&beta;<sub>V4</sub> = -0.518 published vs +0.704 empirical).
            </p>
            <p>
              <strong>Outcome:</strong> Evaluating published Table 5 coefficients on the genuine dataset generated 46,687 false positives (Precision: 0.15%, ROC-AUC: 0.3769).
            </p>
          </div>
        </div>

        {/* Card 2: Guided Fuzzy Learning Formulation */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <CheckCircle2 className="w-4 h-4" />
            <span>2. Guided Fuzzy Learning Innovation</span>
          </div>
          <div className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <p>
              <strong>TFN Parameterization:</strong> Each coefficient is modeled as a Triangular Fuzzy Number A&#771;<sub>j</sub> = (a<sub>j</sub>, b<sub>j</sub>, c<sub>j</sub>) parameterized by non-negative semi-spreads a<sub>j</sub> = b<sub>j</sub> - s<sub>l,j</sub>, c<sub>j</sub> = b<sub>j</sub> + s<sub>r,j</sub> with s &ge; 0.005, strictly guaranteeing a<sub>j</sub> &le; b<sub>j</sub> &le; c<sub>j</sub>.
            </p>
            <p>
              <strong>Fisher Information Initialization:</strong> Spreads are initialized from the asymptotic standard errors of the inverse Fisher information matrix I<sup>-1</sup>(&beta;) on training data.
            </p>
            <p>
              <strong>Performance Gain:</strong> PR-AUC surged from <strong>0.0310 &rarr; 0.6972</strong> (&gt;22&times; baseline prevalence), and ROC-AUC improved from <strong>0.6900 &rarr; 0.9810</strong>, catching 64 of 75 frauds.
            </p>
          </div>
        </div>

        {/* Card 3: Prequential Streaming Protocol & Delayed Labels */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md space-y-3">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-sm">
            <Clock className="w-4 h-4" />
            <span>3. Prequential Evaluation Protocol</span>
          </div>
          <div className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <p>
              <strong>Test-Then-Train Ordering:</strong> Every transaction is scored strictly before its label is placed into the pending maturity queue. Zero lookahead leakage.
            </p>
            <p>
              <strong>1-Hour Label Delay (&Delta;t = 3600s):</strong> Models feedback latency where chargeback dispute signals mature only after 1 hour of simulation time.
            </p>
            <p>
              <strong>Adaptive Gain:</strong> Under 1-hour delay, false alarms decrease from <strong>267 down to 204</strong> (a 23.6% reduction) while preserving full 85.33% recall.
            </p>
          </div>
        </div>

        {/* Card 4: Uncertainty Semantics: Fuzzy Bounds vs Statistical CIs */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md space-y-3">
          <div className="flex items-center gap-2 text-amber-400 font-semibold text-sm">
            <Scale className="w-4 h-4" />
            <span>4. Uncertainty Semantics Disclosure</span>
          </div>
          <div className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <p>
              <strong>Fuzzy Probability Bounds:</strong> Triangular fuzzy numbers generate lower and upper probability bounds [p<sub>left</sub>, p<sub>right</sub>] via possibility theory and Dubois&ndash;Prade interval arithmetic.
            </p>
            <p>
              <strong>Not Statistical Confidence Intervals:</strong> These bounds have no frequentist repeated-sampling coverage proof or Bayesian posterior derivation. Discrete binary labels y &isin; &#123;0, 1&#125; cannot be contained in open sub-intervals of (0, 1).
            </p>
            <p>
              <strong>Significance:</strong> Fuzzy bounds reflect model parameter sensitivity and coefficient epistemic ambiguity, not calibrated probability confidence.
            </p>
          </div>
        </div>
      </div>

      {/* Critical Limitations Box */}
      <div className="p-6 rounded-xl bg-slate-900 border border-rose-900/50 shadow-md space-y-4">
        <h3 className="text-sm font-bold text-rose-300 flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-rose-400" />
          Formal Methodological Limitations & Deployment Boundaries
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300">
          <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-1.5">
            <h4 className="font-semibold text-white">Short 48-Hour Historical Span</h4>
            <p className="text-slate-400 leading-relaxed">
              The dataset contains exactly 172,792 seconds (~2 days) of credit card activity from September 2013. While micro-drift across 7.65 hours of test transactions is demonstrated, macroeconomic macro-drift (seasonal trends, holiday spikes, merchant category shifts) cannot be validated on a 2-day historical trace.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-1.5">
            <h4 className="font-semibold text-white">Realistic Banking Chargeback Horizon</h4>
            <p className="text-slate-400 leading-relaxed">
              In real retail banking, formal chargebacks take 30 to 90 days to settle through card networks (Visa/Mastercard dispute cycles). The simulated 1-hour delay models automated high-velocity customer alerts, not full chargeback resolution.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-1.5">
            <h4 className="font-semibold text-white">Oracle Review Simulation Assumption</h4>
            <p className="text-slate-400 leading-relaxed">
              Reported false-positive reductions in human triage assume an <em>oracle investigator</em> with 100% classification accuracy. In live banking operations, human fraud analysts have finite review capacity, fatigue, and non-zero false positive error rates.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-1.5">
            <h4 className="font-semibold text-white">Replay Memory Requirement</h4>
            <p className="text-slate-400 leading-relaxed">
              Streaming adaptation on 0.13% class imbalance requires a historical replay reservoir to anchor positive class representation. Pure memory-less streaming gradient descent collapses due to catastrophic forgetting of rare fraud events.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

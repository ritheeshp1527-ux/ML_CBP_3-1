import React, { useState } from 'react';
import { 
  Users, 
  AlertTriangle, 
  CheckCircle2, 
  HelpCircle, 
  Sliders, 
  BarChart3, 
  ShieldAlert 
} from 'lucide-react';

export default function TriageTab({ triage }) {
  const [selectedBudget, setSelectedBudget] = useState(2.0); // 0.5, 1.0, 2.0, 5.0
  const [selectedProtocol, setSelectedProtocol] = useState('validation_calibrated_cutoff'); // validation_calibrated_cutoff vs stream_rank_cutoff

  if (!triage || !triage.data) {
    return (
      <div className="py-20 text-center text-slate-400">
        <div className="inline-block animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full mb-3"></div>
        <p>Loading human-review triage results...</p>
      </div>
    );
  }

  const allRecords = triage.data;

  // Filter records for selected budget and protocol
  const marginRecord = allRecords.find(
    r => r.strategy === 'margin_uncertainty_triage' && 
         r.protocol === selectedProtocol && 
         r.target_budget_pct === selectedBudget
  );

  const widthRecord = allRecords.find(
    r => r.strategy === 'fuzzy_width_triage' && 
         r.protocol === selectedProtocol && 
         r.target_budget_pct === selectedBudget
  );

  const budgetOptions = [0.5, 1.0, 2.0, 5.0];

  return (
    <div className="space-y-6">
      {/* Header & Oracle Review Disclosure Banner */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Users className="w-5 h-5 text-cyan-400" />
              Human-Review Triage Policy Evaluation
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Decision-support simulation routing ambiguous transactions to human investigation
            </p>
          </div>

          {/* Protocol Toggle */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 p-1 rounded-lg border border-slate-700">
            <span className="text-[11px] text-slate-400 px-2 font-medium">Protocol:</span>
            <button
              onClick={() => setSelectedProtocol('validation_calibrated_cutoff')}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                selectedProtocol === 'validation_calibrated_cutoff'
                  ? 'bg-cyan-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Validation-Calibrated (Out-of-Sample)
            </button>
            <button
              onClick={() => setSelectedProtocol('stream_rank_cutoff')}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                selectedProtocol === 'stream_rank_cutoff'
                  ? 'bg-cyan-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Stream-Rank Cutoff (In-Sample)
            </button>
          </div>
        </div>

        {/* Oracle Simulation Disclosure Notice */}
        <div className="p-3.5 rounded-lg bg-amber-950/40 border border-amber-800/70 text-xs text-amber-200/90 flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <strong className="text-amber-300 font-semibold">Important Scientific Disclosure (Oracle-Review Simulation):</strong>
            <p>
              The reported post-review false-positive reductions assume an <em>oracle human investigator</em> who correctly resolves 100% of reviewed transactions. This is a mathematical decision-support simulation, not empirical evidence of human accuracy in live banking operations.
            </p>
          </div>
        </div>
      </div>

      {/* Review Budget Selector */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-white">Select Investigation Review Budget:</span>
        </div>

        <div className="flex items-center gap-2">
          {budgetOptions.map(b => (
            <button
              key={b}
              onClick={() => setSelectedBudget(b)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                selectedBudget === b
                  ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {b}% Budget
            </button>
          ))}
        </div>
      </div>

      {/* Strategy Comparison Cards: Margin vs Fuzzy Width */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Margin Uncertainty Triage (Recommended) */}
        <div className="p-6 rounded-xl bg-slate-900 border border-cyan-500/40 shadow-lg shadow-cyan-950/20 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Threshold-Margin Triage (Recommended)
              </h3>
            </div>
            <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
              |E(P) - 0.90| &le; Cutoff
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Routes transactions closest to the decision threshold where classification boundary uncertainty is highest.
          </p>

          {marginRecord && (
            <div className="space-y-3 pt-1">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 text-[10px]">Reviewed Queue Size</span>
                  <p className="text-lg font-bold font-mono text-white mt-0.5">
                    {marginRecord.actual_reviewed_count.toLocaleString()} txs
                  </p>
                  <span className="text-[10px] text-slate-400">{marginRecord.actual_reviewed_pct.toFixed(2)}% of stream</span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 text-[10px]">Frauds in Review Queue</span>
                  <p className="text-lg font-bold font-mono text-cyan-400 mt-0.5">
                    {marginRecord.reviewed_frauds} / 75
                  </p>
                  <span className="text-[10px] text-cyan-400/80">{(marginRecord.fraud_recall_in_review * 100).toFixed(1)}% recall</span>
                </div>
              </div>

              {/* Exact Breakdown: Prior TP vs Salvaged FN */}
              <div className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800 text-xs space-y-1.5 font-mono">
                <div className="flex justify-between text-slate-300">
                  <span>Reviewed Model Alerts (Prior TP):</span>
                  <span className="text-white font-bold">{marginRecord.reviewed_prior_tp}</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Salvaged Missed Frauds (Salvaged FN):</span>
                  <span className="text-emerald-400 font-bold">{marginRecord.reviewed_salvaged_fn}</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>False Alarms Cleared by Review (Cleared FP):</span>
                  <span className="text-cyan-400 font-bold">{marginRecord.reviewed_cleared_fp}</span>
                </div>
              </div>

              {/* Remaining Automated Errors Outside Review */}
              <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 text-xs space-y-1">
                <span className="text-[11px] font-semibold text-slate-300">Remaining Automated Errors (Post-Review):</span>
                <div className="flex justify-between text-slate-400 font-mono text-[11px]">
                  <span>Remaining False Negatives: <strong className="text-amber-400">{marginRecord.oracle_auto_false_negatives}</strong></span>
                  <span>Remaining False Positives: <strong className="text-rose-400">{marginRecord.oracle_auto_false_positives}</strong></span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Fuzzy Width Triage */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider">
              Fuzzy Width Triage
            </h3>
            <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              (p_right - p_left) &ge; Cutoff
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Routes transactions with largest coefficient parameter uncertainty across all feature dimensions.
          </p>

          {widthRecord && (
            <div className="space-y-3 pt-1">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 text-[10px]">Reviewed Queue Size</span>
                  <p className="text-lg font-bold font-mono text-white mt-0.5">
                    {widthRecord.actual_reviewed_count.toLocaleString()} txs
                  </p>
                  <span className="text-[10px] text-slate-400">{widthRecord.actual_reviewed_pct.toFixed(2)}% of stream</span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 text-[10px]">Frauds in Review Queue</span>
                  <p className="text-lg font-bold font-mono text-slate-300 mt-0.5">
                    {widthRecord.reviewed_frauds} / 75
                  </p>
                  <span className="text-[10px] text-slate-400">{(widthRecord.fraud_recall_in_review * 100).toFixed(1)}% recall</span>
                </div>
              </div>

              {/* Exact Breakdown: Prior TP vs Salvaged FN */}
              <div className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800 text-xs space-y-1.5 font-mono">
                <div className="flex justify-between text-slate-400">
                  <span>Reviewed Model Alerts (Prior TP):</span>
                  <span className="text-slate-300 font-bold">{widthRecord.reviewed_prior_tp}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Salvaged Missed Frauds (Salvaged FN):</span>
                  <span className="text-slate-300 font-bold">{widthRecord.reviewed_salvaged_fn}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>False Alarms Cleared by Review (Cleared FP):</span>
                  <span className="text-slate-300 font-bold">{widthRecord.reviewed_cleared_fp}</span>
                </div>
              </div>

              {/* Remaining Automated Errors Outside Review */}
              <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 text-xs space-y-1">
                <span className="text-[11px] font-semibold text-slate-300">Remaining Automated Errors (Post-Review):</span>
                <div className="flex justify-between text-slate-400 font-mono text-[11px]">
                  <span>Remaining False Negatives: <strong className="text-amber-400">{widthRecord.oracle_auto_false_negatives}</strong></span>
                  <span>Remaining False Positives: <strong className="text-rose-400">{widthRecord.oracle_auto_false_positives}</strong></span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Critical Scientific Finding Explanation Box */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
        <h4 className="font-semibold text-white flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-cyan-400" />
          Why Threshold-Margin Triage Overwhelmingly Outperforms Pure Fuzzy Width
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-slate-400 pt-1 leading-relaxed">
          <p>
            1. <strong>Sigmoid Derivative Peak:</strong> The logistic mapping derivative &sigma;'(z) reaches its mathematical maximum at z=0 (probability 0.50). Consequently, fuzzy spread width naturally concentrates in the intermediate ambiguous zone (P &in; [0.20, 0.60]), not at the extreme decision threshold &tau;=0.90.
          </p>
          <p>
            2. <strong>Alert Verification Function:</strong> At a 2% review budget, margin triage clears 100% of false alerts because it inspects the model's highest-scoring border alerts. 64 of 65 reviewed frauds were already detected true positives; human review serves as verification rather than unseen fraud discovery.
          </p>
        </div>
      </div>
    </div>
  );
}

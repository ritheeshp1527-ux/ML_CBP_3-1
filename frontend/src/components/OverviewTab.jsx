import React, { useState } from 'react';
import { 
  Database, 
  AlertCircle, 
  CheckCircle2, 
  Clock, 
  TrendingUp, 
  Cpu, 
  ArrowRight,
  ShieldAlert,
  Zap
} from 'lucide-react';

export default function OverviewTab({ summary, models, onSelectTab }) {
  const [selectedStrategy, setSelectedStrategy] = useState('guided_fuzzy_adaptive_delayed_1h');

  if (!summary) {
    return (
      <div className="py-20 text-center text-slate-400">
        <div className="inline-block animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full mb-3"></div>
        <p>Loading research overview and dataset metadata...</p>
      </div>
    );
  }

  const ds = summary.dataset;
  const parts = summary.partitions;

  const currentModelStats = models?.data?.find(m => m.model_key === selectedStrategy) || 
    models?.data?.find(m => m.model_key === 'guided_fuzzy_frozen');

  return (
    <div className="space-y-6">
      {/* Top Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Transactions</span>
            <Database className="w-5 h-5 text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white font-mono">{ds.total_rows.toLocaleString()}</span>
            <span className="text-xs text-slate-400">31 columns</span>
          </div>
          <p className="mt-1 text-xs text-slate-500">Chronological 48-hour credit card trace</p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Fraud Cases (Class 1)</span>
            <ShieldAlert className="w-5 h-5 text-rose-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold text-rose-400 font-mono">{ds.total_frauds}</span>
            <span className="text-xs text-rose-500/80">0.173% dataset rate</span>
          </div>
          <p className="mt-1 text-xs text-slate-500">Severe real-world class imbalance</p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Test Stream Duration</span>
            <Clock className="w-5 h-5 text-indigo-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white font-mono">{parts.test.evaluation_period_hours.toFixed(2)}h</span>
            <span className="text-xs text-indigo-400">75 test frauds</span>
          </div>
          <p className="mt-1 text-xs text-slate-500">Rows 227,845 to 284,806 (56,962 txs)</p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Evaluation Methodology</span>
            <Cpu className="w-5 h-5 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-lg font-bold text-emerald-400">Prequential Stream</span>
          </div>
          <p className="mt-1 text-xs text-slate-500">Test-Then-Train with 1-hr label delay</p>
        </div>
      </div>

      {/* Primary Highlight Banner & Selected Strategy Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 p-6 rounded-xl bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950/40 border border-slate-800 shadow-md space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-cyan-400" />
                Active Model Performance Benchmark
              </h2>
              <p className="text-xs text-slate-400">
                Prequential test-stream metrics on 56,962 chronological transactions
              </p>
            </div>
            {/* Strategy selector dropdown */}
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-cyan-500 font-medium"
            >
              <option value="guided_fuzzy_adaptive_delayed_1h">Guided Fuzzy (Adaptive, 1-hr Delay)</option>
              <option value="guided_fuzzy_adaptive_immediate">Guided Fuzzy (Adaptive, 0s Delay)</option>
              <option value="guided_fuzzy_frozen">Guided Fuzzy (Frozen Model)</option>
              <option value="logistic_regression_frozen">Logistic Regression (Frozen Baseline)</option>
              <option value="logistic_regression_adaptive_immediate">Logistic Regression (Adaptive Baseline)</option>
            </select>
          </div>

          {currentModelStats && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 pt-2">
              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-[11px] text-slate-400">Fraud Recall</span>
                <p className="text-xl font-bold text-cyan-400 font-mono mt-1">
                  {(currentModelStats.recall * 100).toFixed(2)}%
                </p>
                <span className="text-[10px] text-slate-500">{currentModelStats.tp} / 75 caught</span>
              </div>

              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-[11px] text-slate-400">Precision</span>
                <p className="text-xl font-bold text-white font-mono mt-1">
                  {(currentModelStats.precision * 100).toFixed(2)}%
                </p>
                <span className="text-[10px] text-slate-500">{currentModelStats.fp} false alarms</span>
              </div>

              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-[11px] text-slate-400">F1 Score</span>
                <p className="text-xl font-bold text-emerald-400 font-mono mt-1">
                  {currentModelStats.f1.toFixed(4)}
                </p>
                <span className="text-[10px] text-slate-500">Harmonic mean</span>
              </div>

              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-[11px] text-slate-400">PR-AUC</span>
                <p className="text-xl font-bold text-indigo-400 font-mono mt-1">
                  {currentModelStats.pr_auc.toFixed(4)}
                </p>
                <span className="text-[10px] text-slate-500">Baseline: 0.0013</span>
              </div>

              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-[11px] text-slate-400">ROC-AUC</span>
                <p className="text-xl font-bold text-white font-mono mt-1">
                  {currentModelStats.roc_auc.toFixed(4)}
                </p>
                <span className="text-[10px] text-slate-500">Ranking quality</span>
              </div>

              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                <span className="text-[11px] text-slate-400">Inference Latency</span>
                <p className="text-xl font-bold text-amber-400 font-mono mt-1">
                  {(currentModelStats.latency_ms || 0.035).toFixed(3)} ms
                </p>
                <span className="text-[10px] text-slate-500">Per transaction</span>
              </div>
            </div>
          )}

          <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 flex items-center justify-between">
            <div>
              <span className="text-cyan-400 font-semibold">Key Finding:</span> Online adaptation with a 1-hour feedback delay cuts false positives by <strong>23.6% (267 down to 204)</strong> while preserving the exact same <strong>85.33% fraud recall</strong>.
            </div>
            <button
              onClick={() => onSelectTab('streaming')}
              className="flex items-center gap-1 text-cyan-400 hover:text-cyan-300 font-medium pl-3 whitespace-nowrap"
            >
              Analyze Drift <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Dataset Chronological Split Details */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md space-y-4">
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            Chronological Split Protocol
          </h3>
          <p className="text-xs text-slate-400">
            Strict 60 / 20 / 20 split strictly ordered by transaction timestamp. Zero data shuffling or lookahead leakage.
          </p>

          <div className="space-y-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-800">
              <div className="flex justify-between font-semibold text-slate-200">
                <span>Training Partition (60%)</span>
                <span className="font-mono text-cyan-400">170,884 txs</span>
              </div>
              <div className="mt-1 flex justify-between text-slate-400">
                <span>360 frauds (0.211%)</span>
                <span>T: 0.0s – 120,396s (~33.4h)</span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-800">
              <div className="flex justify-between font-semibold text-slate-200">
                <span>Validation Partition (20%)</span>
                <span className="font-mono text-amber-400">56,961 txs</span>
              </div>
              <div className="mt-1 flex justify-between text-slate-400">
                <span>57 frauds (0.100%)</span>
                <span>T: 120,396s – 145,247s (~6.9h)</span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-800">
              <div className="flex justify-between font-semibold text-slate-200">
                <span>Test Stream Partition (20%)</span>
                <span className="font-mono text-emerald-400">56,962 txs</span>
              </div>
              <div className="mt-1 flex justify-between text-slate-400">
                <span>75 frauds (0.132%)</span>
                <span>T: 145,248s – 172,792s (~7.65h)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div 
          onClick={() => onSelectTab('comparison')}
          className="p-4 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/50 cursor-pointer transition-all group"
        >
          <div className="flex items-center justify-between text-slate-300 font-semibold text-sm">
            <span>Model Comparison</span>
            <ArrowRight className="w-4 h-4 text-cyan-400 group-hover:translate-x-1 transition-transform" />
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Compare 11 offline baselines and streaming models across Precision, Recall, F1, PR-AUC, and latency.
          </p>
        </div>

        <div 
          onClick={() => onSelectTab('triage')}
          className="p-4 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/50 cursor-pointer transition-all group"
        >
          <div className="flex items-center justify-between text-slate-300 font-semibold text-sm">
            <span>Human-Review Triage</span>
            <ArrowRight className="w-4 h-4 text-cyan-400 group-hover:translate-x-1 transition-transform" />
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Explore decision support at 0.5%–5% review budgets. Evaluates oracle review simulation trade-offs.
          </p>
        </div>

        <div 
          onClick={() => onSelectTab('explanation')}
          className="p-4 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/50 cursor-pointer transition-all group"
        >
          <div className="flex items-center justify-between text-slate-300 font-semibold text-sm">
            <span>Fuzzy Explanation Tool</span>
            <ArrowRight className="w-4 h-4 text-cyan-400 group-hover:translate-x-1 transition-transform" />
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Inspect individual transactions or test synthetic features with defuzzified bounds [p_left, p_right].
          </p>
        </div>
      </div>
    </div>
  );
}

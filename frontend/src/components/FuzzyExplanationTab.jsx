import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  Search, 
  AlertTriangle, 
  Info, 
  Sparkles, 
  Sliders, 
  HelpCircle,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import { fetchTransactionExplanation, predictCustomTransaction } from '../api';

export default function FuzzyExplanationTab() {
  const [txIdInput, setTxIdInput] = useState('541'); // Default to a known fraud in dataset
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [txData, setTxData] = useState(null);

  // Custom simulator state
  const [activeMode, setActiveMode] = useState('inspector'); // inspector vs simulator
  const [customAmount, setCustomAmount] = useState('249.99');
  const [customV1, setCustomV1] = useState('-2.3');
  const [customV4, setCustomV4] = useState('3.8');
  const [customV14, setCustomV14] = useState('-4.2');
  const [customResult, setCustomResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);

  // Fetch initial explanation
  useEffect(() => {
    handleLookup('541');
  }, []);

  async function handleLookup(idToLookup) {
    const id = idToLookup || txIdInput;
    if (!id.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchTransactionExplanation(parseInt(id, 10));
      setTxData(data);
    } catch (err) {
      setError(err.message || 'Failed to load transaction');
    } finally {
      setLoading(false);
    }
  }

  async function handleScoreCustom() {
    setSimLoading(true);
    try {
      const res = await predictCustomTransaction({
        Time: 150000.0,
        Amount: parseFloat(customAmount) || 0.0,
        V1: parseFloat(customV1) || 0.0,
        V4: parseFloat(customV4) || 0.0,
        V14: parseFloat(customV14) || 0.0,
        is_synthetic: true,
      });
      setCustomResult(res);
    } catch (err) {
      alert('Error scoring custom transaction: ' + err.message);
    } finally {
      setSimLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header and Mode Switcher */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-cyan-400" />
            Fuzzy Uncertainty & Transaction Explanation
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Explain epistemic parameter ambiguity, triangular fuzzy bounds [p_left, p_right], and feature attributions
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-800/80 p-1 rounded-lg border border-slate-700">
          <button
            onClick={() => setActiveMode('inspector')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              activeMode === 'inspector' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Dataset Transaction Inspector
          </button>
          <button
            onClick={() => setActiveMode('simulator')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              activeMode === 'simulator' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Synthetic Feature Simulator
          </button>
        </div>
      </div>

      {activeMode === 'inspector' ? (
        <div className="space-y-6">
          {/* Lookup Input Bar */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-300 font-medium">Inspect Transaction ID:</span>
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  value={txIdInput}
                  onChange={(e) => setTxIdInput(e.target.value)}
                  placeholder="e.g. 541, 0, 1200"
                  className="bg-slate-950 border border-slate-700 text-slate-100 text-xs rounded-lg px-3 py-1.5 w-36 font-mono focus:outline-none focus:ring-1 focus:ring-cyan-500"
                />
                <button
                  onClick={() => handleLookup()}
                  disabled={loading}
                  className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors"
                >
                  <Search className="w-3.5 h-3.5" />
                  {loading ? 'Inspecting...' : 'Analyze'}
                </button>
              </div>
            </div>

            {/* Quick preset buttons */}
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>Presets:</span>
              <button 
                onClick={() => { setTxIdInput('541'); handleLookup('541'); }}
                className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-rose-300 font-mono text-[11px]"
              >
                ID 541 (Fraud)
              </button>
              <button 
                onClick={() => { setTxIdInput('623'); handleLookup('623'); }}
                className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-rose-300 font-mono text-[11px]"
              >
                ID 623 (Fraud)
              </button>
              <button 
                onClick={() => { setTxIdInput('0'); handleLookup('0'); }}
                className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 font-mono text-[11px]"
              >
                ID 0 (Legit)
              </button>
              <button 
                onClick={() => { setTxIdInput('100'); handleLookup('100'); }}
                className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 font-mono text-[11px]"
              >
                ID 100 (Legit)
              </button>
            </div>
          </div>

          {error && (
            <div className="p-4 rounded-xl bg-rose-950/50 border border-rose-800 text-rose-200 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          {txData && (
            <div className="space-y-6">
              {/* Primary Fuzzy Metrics Panel */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {/* Expected Probability Card */}
                <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Defuzzified Probability</span>
                    <HelpCircle className="w-3.5 h-3.5 text-slate-500" title="Dubois & Prade expected value" />
                  </div>
                  <div className="text-3xl font-extrabold font-mono text-cyan-400">
                    {(txData.expected_value * 100).toFixed(2)}%
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Decision Threshold: <span className="font-mono text-slate-300">{(txData.threshold * 100).toFixed(1)}%</span>
                  </div>
                </div>

                {/* Probability Interval Bounds */}
                <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Fuzzy Bounds [p_left, p_right]</span>
                  </div>
                  <div className="text-base font-bold font-mono text-slate-200 flex items-center gap-1">
                    <span>[{(txData.fuzzy_bounds.p_left * 100).toFixed(1)}%,</span>
                    <span className="text-cyan-400 font-bold">{(txData.fuzzy_bounds.p_center * 100).toFixed(1)}%,</span>
                    <span>{(txData.fuzzy_bounds.p_right * 100).toFixed(1)}%]</span>
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Interval width: <span className="font-mono text-indigo-300">{(txData.fuzzy_bounds.uncertainty_width * 100).toFixed(2)}%</span>
                  </div>
                </div>

                {/* Classification Verdict */}
                <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Model Verdict</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2.5 py-1 rounded text-sm font-bold tracking-wider uppercase ${
                      txData.predicted_class === 1
                        ? 'bg-rose-950 text-rose-300 border border-rose-800'
                        : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                    }`}>
                      {txData.classification_label}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Ground truth: <span className="font-mono font-semibold text-slate-300">{txData.true_label === 1 ? 'Fraud (1)' : 'Legitimate (0)'}</span>
                  </div>
                </div>

                {/* Decision Margin */}
                <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Threshold Distance</span>
                  </div>
                  <div className="text-2xl font-bold font-mono text-white">
                    {txData.fuzzy_bounds.margin_to_threshold.toFixed(4)}
                  </div>
                  <div className="text-[11px] text-slate-400">
                    {txData.fuzzy_bounds.margin_to_threshold <= 0.24398 ? (
                      <span className="text-amber-400 font-medium">Routed to Review (High Ambiguity)</span>
                    ) : (
                      <span className="text-slate-500">Auto-Decided (Low Ambiguity)</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Visual Probability Bracket Bar */}
              <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-200">Fuzzy Interval Spread Visualization</span>
                  <span className="text-slate-400 font-mono">0.0 (Legitimate) to 1.0 (Fraud)</span>
                </div>

                {/* Meter container */}
                <div className="relative h-8 bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
                  {/* Left bound to right bound highlighted area */}
                  <div 
                    className="absolute top-0 bottom-0 bg-cyan-500/25 border-l-2 border-r-2 border-cyan-400"
                    style={{
                      left: `${Math.min(txData.fuzzy_bounds.p_left * 100, 100)}%`,
                      width: `${Math.max((txData.fuzzy_bounds.p_right - txData.fuzzy_bounds.p_left) * 100, 1)}%`
                    }}
                  />
                  {/* Center probability marker */}
                  <div 
                    className="absolute top-0 bottom-0 w-1 bg-cyan-400 shadow-sm shadow-cyan-400/50"
                    style={{ left: `${txData.fuzzy_bounds.p_center * 100}%` }}
                    title={`Center probability: ${(txData.fuzzy_bounds.p_center * 100).toFixed(2)}%`}
                  />
                  {/* Decision threshold line */}
                  <div 
                    className="absolute top-0 bottom-0 w-0.5 bg-rose-500 border-dashed border-rose-400 z-10"
                    style={{ left: `${txData.threshold * 100}%` }}
                    title={`Threshold: ${(txData.threshold * 100).toFixed(1)}%`}
                  />
                </div>

                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>p_left: {(txData.fuzzy_bounds.p_left * 100).toFixed(2)}%</span>
                  <span className="text-cyan-400 font-semibold">Expected: {(txData.expected_value * 100).toFixed(2)}%</span>
                  <span className="text-rose-400">Threshold: {(txData.threshold * 100).toFixed(1)}%</span>
                  <span>p_right: {(txData.fuzzy_bounds.p_right * 100).toFixed(2)}%</span>
                </div>
              </div>

              {/* Top Feature Attributions */}
              <div className="rounded-xl bg-slate-900 border border-slate-800 shadow-md overflow-hidden">
                <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">Top 10 Feature Contributions (Signed TFN Impact)</h3>
                  <span className="text-xs text-slate-400">Transaction Amount: ${txData.amount.toFixed(2)}</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left text-slate-300">
                    <thead className="text-[11px] uppercase tracking-wider bg-slate-950/60 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="py-2.5 px-4 font-semibold">Feature</th>
                        <th className="py-2.5 px-3 font-semibold text-right">Raw Value</th>
                        <th className="py-2.5 px-3 font-semibold text-right">TFN Center (b)</th>
                        <th className="py-2.5 px-3 font-semibold text-right">Center Contribution</th>
                        <th className="py-2.5 px-3 font-semibold text-right">Fuzzy Spread Impact</th>
                        <th className="py-2.5 px-3 font-semibold text-center">Direction</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 font-mono text-[11px]">
                      {txData.top_features.map((f, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                          <td className="py-2.5 px-4 font-sans font-medium text-slate-200">{f.feature}</td>
                          <td className="py-2.5 px-3 text-right text-slate-400">{f.raw_value.toFixed(4)}</td>
                          <td className="py-2.5 px-3 text-right text-slate-300">{f.tfn_b.toFixed(4)}</td>
                          <td className={`py-2.5 px-3 text-right font-bold ${
                            f.center_contribution > 0 ? 'text-rose-400' : 'text-emerald-400'
                          }`}>
                            {f.center_contribution > 0 ? '+' : ''}{f.center_contribution.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-right text-indigo-300">
                            ±{f.spread_contribution.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-center font-sans">
                            {f.is_risk_increasing ? (
                              <span className="inline-flex items-center gap-1 text-[10px] text-rose-400 bg-rose-950/40 px-2 py-0.5 rounded border border-rose-900">
                                <TrendingUp className="w-3 h-3" /> Risk +
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-900">
                                <TrendingDown className="w-3 h-3" /> Risk -
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Synthetic Feature Simulator View */
        <div className="space-y-6">
          <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-800 text-xs text-indigo-200 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400 shrink-0" />
            <span>
              <strong>Synthetic Simulator Notice:</strong> Test arbitrary or extreme fraud scenarios using the trained Guided Fuzzy Logistic Regression pipeline. Results are clearly tagged as synthetic.
            </span>
          </div>

          <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md space-y-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Adjust Synthetic Feature Inputs
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div className="space-y-1">
                <label className="text-slate-400 font-medium">Transaction Amount ($)</label>
                <input
                  type="number"
                  value={customAmount}
                  onChange={(e) => setCustomAmount(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 font-mono text-xs focus:ring-1 focus:ring-cyan-500"
                />
              </div>

              <div className="space-y-1">
                <label className="text-slate-400 font-medium">Feature V4 (High Risk Driver)</label>
                <input
                  type="number"
                  value={customV4}
                  onChange={(e) => setCustomV4(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 font-mono text-xs focus:ring-1 focus:ring-cyan-500"
                />
              </div>

              <div className="space-y-1">
                <label className="text-slate-400 font-medium">Feature V14 (Inverse Risk Driver)</label>
                <input
                  type="number"
                  value={customV14}
                  onChange={(e) => setCustomV14(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 font-mono text-xs focus:ring-1 focus:ring-cyan-500"
                />
              </div>

              <div className="space-y-1">
                <label className="text-slate-400 font-medium">Feature V1</label>
                <input
                  type="number"
                  value={customV1}
                  onChange={(e) => setCustomV1(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 font-mono text-xs focus:ring-1 focus:ring-cyan-500"
                />
              </div>
            </div>

            <div className="pt-2">
              <button
                onClick={handleScoreCustom}
                disabled={simLoading}
                className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold flex items-center gap-2 transition-colors"
              >
                <Sparkles className="w-4 h-4" />
                {simLoading ? 'Simulating...' : 'Run Fuzzy Scoring'}
              </button>
            </div>

            {customResult && (
              <div className="mt-6 p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">
                    Synthetic Transaction Score Outcome
                  </span>
                  <span className={`px-2.5 py-0.5 rounded text-xs font-bold ${
                    customResult.predicted_class === 1 
                      ? 'bg-rose-950 text-rose-300 border border-rose-800' 
                      : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                  }`}>
                    {customResult.verdict}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-2">
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 text-[10px]">Expected Probability</span>
                    <p className="text-lg font-bold font-mono text-cyan-400 mt-1">
                      {(customResult.expected_value * 100).toFixed(2)}%
                    </p>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 text-[10px]">Fuzzy Interval</span>
                    <p className="text-xs font-bold font-mono text-slate-200 mt-1">
                      [{(customResult.p_left * 100).toFixed(1)}%, {(customResult.p_right * 100).toFixed(1)}%]
                    </p>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 text-[10px]">Fuzzy Width</span>
                    <p className="text-lg font-bold font-mono text-indigo-300 mt-1">
                      {(customResult.uncertainty_width * 100).toFixed(2)}%
                    </p>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 text-[10px]">Review Triage</span>
                    <p className={`text-xs font-bold mt-1 ${
                      customResult.routed_to_review ? 'text-amber-400' : 'text-slate-400'
                    }`}>
                      {customResult.routed_to_review ? 'Flagged for Review' : 'Auto Processed'}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

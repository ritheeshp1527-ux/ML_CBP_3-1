import React, { useState } from 'react';
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend 
} from 'recharts';
import { Layers, Clock, TrendingUp, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function StreamingAdaptationTab({ windows }) {
  const [activeStrategyFilter, setActiveStrategyFilter] = useState('all');

  if (!windows || !windows.data) {
    return (
      <div className="py-20 text-center text-slate-400">
        <div className="inline-block animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full mb-3"></div>
        <p>Loading streaming adaptation and concept drift metrics...</p>
      </div>
    );
  }

  const allWindows = windows.data;
  
  // Format window drift line chart data across Windows 1..4
  const windowIds = [1, 2, 3, 4];
  const driftChartData = windowIds.map(wId => {
    const rowFrozen = allWindows.find(d => d.window_id === wId && d.strategy === 'guided_fuzzy_frozen');
    const rowAdaptiveImm = allWindows.find(d => d.window_id === wId && d.strategy === 'guided_fuzzy_adaptive_immediate');
    const rowAdaptiveDel = allWindows.find(d => d.window_id === wId && d.strategy === 'guided_fuzzy_adaptive_delayed_1h');
    const rowLRFrozen = allWindows.find(d => d.window_id === wId && d.strategy === 'logistic_regression_frozen');
    const rowLRAdaptive = allWindows.find(d => d.window_id === wId && d.strategy === 'logistic_regression_adaptive_immediate');

    return {
      window: `Window ${wId}`,
      timeHours: rowFrozen ? `${rowFrozen.duration_hours.toFixed(1)}h` : '',
      frauds: rowFrozen ? rowFrozen.fraud_count : 0,
      fuzzyFrozenRecall: rowFrozen ? Number((rowFrozen.recall * 100).toFixed(1)) : 0,
      fuzzyAdaptiveDelayedRecall: rowAdaptiveDel ? Number((rowAdaptiveDel.recall * 100).toFixed(1)) : 0,
      fuzzyAdaptiveImmRecall: rowAdaptiveImm ? Number((rowAdaptiveImm.recall * 100).toFixed(1)) : 0,
      lrFrozenRecall: rowLRFrozen ? Number((rowLRFrozen.recall * 100).toFixed(1)) : 0,
      lrAdaptiveRecall: rowLRAdaptive ? Number((rowLRAdaptive.recall * 100).toFixed(1)) : 0,
      // False Positives
      fuzzyFrozenFP: rowFrozen ? rowFrozen.fp : 0,
      fuzzyAdaptiveDelayedFP: rowAdaptiveDel ? rowAdaptiveDel.fp : 0,
      fuzzyAdaptiveImmFP: rowAdaptiveImm ? rowAdaptiveImm.fp : 0,
    };
  });

  return (
    <div className="space-y-6">
      {/* Header and Explanation */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            Prequential Streaming Adaptation & Drift
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Tracking model stability, false-alarm reduction, and concept drift across 4 chronological windows (~1.9h each)
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300">
          <Clock className="w-4 h-4 text-cyan-400" />
          <span>Label Delays: <strong>0s (Immediate)</strong> vs <strong>3,600s (1-Hour Delay)</strong></span>
        </div>
      </div>

      {/* Two Comparative Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: Fraud Recall Across Windows */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md">
          <h3 className="text-sm font-semibold text-slate-200 mb-2">
            Fraud Recall Stability Across Sequential Windows (%)
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Notice: In Window 4, Frozen Logistic Regression collapses to 0.0% recall, while Guided Fuzzy retains 63.6% recall.
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={driftChartData} margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="window" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line 
                  type="monotone" 
                  dataKey="fuzzyAdaptiveDelayedRecall" 
                  name="Guided Fuzzy (1-hr Delay)" 
                  stroke="#06b6d4" 
                  strokeWidth={2.5} 
                  dot={{ r: 4 }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="fuzzyFrozenRecall" 
                  name="Guided Fuzzy (Frozen)" 
                  stroke="#a855f7" 
                  strokeWidth={2} 
                  strokeDasharray="4 4" 
                  dot={{ r: 4 }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="lrFrozenRecall" 
                  name="LogReg Baseline (Frozen)" 
                  stroke="#f43f5e" 
                  strokeWidth={2} 
                  dot={{ r: 4 }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: False Positive Reduction by Adaptation */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md">
          <h3 className="text-sm font-semibold text-slate-200 mb-2">
            False Positive Count per Window (Lower is Better)
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Online adaptation systematically suppresses false alarms across every chronological window.
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={driftChartData} margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="window" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Bar dataKey="fuzzyFrozenFP" name="Frozen Fuzzy FP" fill="#64748b" radius={[3, 3, 0, 0]} />
                <Bar dataKey="fuzzyAdaptiveDelayedFP" name="Adaptive (1-hr Delay) FP" fill="#06b6d4" radius={[3, 3, 0, 0]} />
                <Bar dataKey="fuzzyAdaptiveImmFP" name="Adaptive (Immediate) FP" fill="#10b981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Key Finding Callouts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs space-y-2">
          <div className="flex items-center gap-2 font-semibold text-cyan-400">
            <TrendingUp className="w-4 h-4" />
            <span>23.6% False Positive Reduction</span>
          </div>
          <p className="text-slate-400">
            Under a realistic 1-hour delay, false alarms decrease from <strong>267 down to 204</strong>, boosting F1 from 0.3153 to <strong>0.3732</strong> without losing any of the 64 caught frauds.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs space-y-2">
          <div className="flex items-center gap-2 font-semibold text-rose-400">
            <AlertTriangle className="w-4 h-4" />
            <span>Baseline Collapse Prevention</span>
          </div>
          <p className="text-slate-400">
            Standard online SGD suffered severe drift, flagging <strong>1,320 false positives</strong>. In contrast, Guided Fuzzy bounded spreads ($s \ge 0.005$) and proximal regularizers ensure stability.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs space-y-2">
          <div className="flex items-center gap-2 font-semibold text-emerald-400">
            <ShieldCheck className="w-4 h-4" />
            <span>Low Computational Overhead</span>
          </div>
          <p className="text-slate-400">
            Total adaptation time across 20 batch updates is only <strong>3.72 seconds</strong>. Per-transaction inference latency remains ultra-fast at <strong>0.035 ms</strong>.
          </p>
        </div>
      </div>

      {/* Detailed Window Breakdown Table */}
      <div className="rounded-xl bg-slate-900 border border-slate-800 shadow-md overflow-hidden">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Chronological Window Performance Log</h3>
          <span className="text-xs text-slate-400">4 Equal-Volume Windows (~14,240 txs each)</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-slate-300">
            <thead className="text-[11px] uppercase tracking-wider bg-slate-950/60 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3 px-4 font-semibold">Strategy</th>
                <th className="py-3 px-3 font-semibold">Window</th>
                <th className="py-3 px-3 font-semibold text-right">Duration</th>
                <th className="py-3 px-3 font-semibold text-right">Frauds</th>
                <th className="py-3 px-3 font-semibold text-right">Recall</th>
                <th className="py-3 px-3 font-semibold text-right">Precision</th>
                <th className="py-3 px-3 font-semibold text-right">F1</th>
                <th className="py-3 px-3 font-semibold text-right">FP</th>
                <th className="py-3 px-3 font-semibold text-right">FN</th>
                <th className="py-3 px-3 font-semibold text-right">Versions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono text-[11px]">
              {allWindows.map((w, idx) => {
                const isHighlight = w.strategy.includes('guided_fuzzy_adaptive_delayed_1h');
                return (
                  <tr key={idx} className={`hover:bg-slate-800/40 transition-colors ${isHighlight ? 'bg-cyan-950/20' : ''}`}>
                    <td className="py-2.5 px-4 font-sans font-medium text-slate-200">{w.strategy}</td>
                    <td className="py-2.5 px-3 font-sans">Window {w.window_id}</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">{w.duration_hours.toFixed(2)}h</td>
                    <td className="py-2.5 px-3 text-right font-semibold text-rose-400">{w.fraud_count}</td>
                    <td className="py-2.5 px-3 text-right font-semibold text-cyan-400">{(w.recall * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-right text-slate-200">{(w.precision * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-right font-bold text-emerald-400">{w.f1.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right text-amber-400 font-semibold">{w.fp}</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">{w.fn}</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">v{w.model_version_min}–v{w.model_version_max}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

import React, { useState } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend 
} from 'recharts';
import { BarChart3, Filter, CheckCircle2, AlertCircle } from 'lucide-react';

export default function ModelComparisonTab({ models }) {
  const [filterType, setFilterType] = useState('all'); // all, static_held_out, prequential_streaming
  const [activeMetric, setActiveMetric] = useState('f1'); // f1, recall, precision, pr_auc, roc_auc

  if (!models || !models.data) {
    return (
      <div className="py-20 text-center text-slate-400">
        <div className="inline-block animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full mb-3"></div>
        <p>Loading model comparison data...</p>
      </div>
    );
  }

  const allModels = models.data;
  const filteredModels = allModels.filter(m => {
    if (filterType === 'all') return true;
    return m.evaluation_type === filterType;
  });

  const metricLabels = {
    f1: 'F1 Score',
    recall: 'Fraud Recall',
    precision: 'Precision',
    pr_auc: 'PR-AUC (Avg Precision)',
    roc_auc: 'ROC-AUC',
    fp: 'False Positives',
    fn: 'False Negatives',
  };

  const chartData = filteredModels.map(m => ({
    name: m.model_key.length > 20 ? m.model_key.slice(0, 18) + '...' : m.model_key,
    fullName: m.display_name,
    f1: Number((m.f1 || 0).toFixed(4)),
    recall: Number((m.recall * 100 || 0).toFixed(2)),
    precision: Number((m.precision * 100 || 0).toFixed(2)),
    pr_auc: Number((m.pr_auc || 0).toFixed(4)),
    roc_auc: Number((m.roc_auc || 0).toFixed(4)),
    fp: m.fp,
    fn: m.fn,
    evalType: m.evaluation_type,
  }));

  return (
    <div className="space-y-6">
      {/* Header and Filter Controls */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            Empirical Model Comparison
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Fair comparative benchmark distinguishing static held-out test partitions from prequential streaming streams
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Evaluation Type Filter */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 p-1 rounded-lg border border-slate-700">
            <span className="text-[11px] text-slate-400 px-2 font-medium">Evaluation:</span>
            <button
              onClick={() => setFilterType('all')}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                filterType === 'all' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              All Models ({allModels.length})
            </button>
            <button
              onClick={() => setFilterType('static_held_out')}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                filterType === 'static_held_out' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Static Held-Out
            </button>
            <button
              onClick={() => setFilterType('prequential_streaming')}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                filterType === 'prequential_streaming' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Prequential Streaming
            </button>
          </div>

          {/* Metric Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 p-1 rounded-lg border border-slate-700">
            <span className="text-[11px] text-slate-400 px-2 font-medium">Metric:</span>
            <select
              value={activeMetric}
              onChange={(e) => setActiveMetric(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded px-2.5 py-1 focus:outline-none focus:ring-1 focus:ring-cyan-500"
            >
              <option value="f1">F1 Score</option>
              <option value="recall">Recall (%)</option>
              <option value="precision">Precision (%)</option>
              <option value="pr_auc">PR-AUC</option>
              <option value="roc_auc">ROC-AUC</option>
            </select>
          </div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-md">
        <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center justify-between">
          <span>Comparative Bar Chart — {metricLabels[activeMetric]}</span>
          <span className="text-xs text-slate-400 font-normal">Higher is better</span>
        </h3>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis 
                dataKey="name" 
                stroke="#64748b" 
                fontSize={11} 
                interval={0} 
                angle={-25} 
                textAnchor="end" 
              />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                formatter={(val) => [val, metricLabels[activeMetric]]}
                labelFormatter={(name) => {
                  const item = chartData.find(d => d.name === name);
                  return item ? item.fullName : name;
                }}
              />
              <Bar 
                dataKey={activeMetric} 
                fill="#06b6d4" 
                radius={[4, 4, 0, 0]} 
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Comprehensive Table */}
      <div className="rounded-xl bg-slate-900 border border-slate-800 shadow-md overflow-hidden">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Full Comparative Evaluation Table</h3>
          <span className="text-xs text-slate-400">Total Evaluated: {filteredModels.length} models</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-slate-300">
            <thead className="text-[11px] uppercase tracking-wider bg-slate-950/60 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3 px-4 font-semibold">Model / Strategy</th>
                <th className="py-3 px-3 font-semibold">Type</th>
                <th className="py-3 px-3 font-semibold">Thresh</th>
                <th className="py-3 px-3 font-semibold text-right">Recall</th>
                <th className="py-3 px-3 font-semibold text-right">Precision</th>
                <th className="py-3 px-3 font-semibold text-right">F1 Score</th>
                <th className="py-3 px-3 font-semibold text-right">PR-AUC</th>
                <th className="py-3 px-3 font-semibold text-right">ROC-AUC</th>
                <th className="py-3 px-3 font-semibold text-right">FP</th>
                <th className="py-3 px-3 font-semibold text-right">FN</th>
                <th className="py-3 px-3 font-semibold text-right">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono text-[11px]">
              {filteredModels.map((m, idx) => {
                const isAdaptive = m.evaluation_type === 'prequential_streaming';
                const isHighlight = m.model_key.includes('guided_fuzzy');
                return (
                  <tr 
                    key={idx} 
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isHighlight ? 'bg-cyan-950/15' : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-slate-200">
                      <div className="flex items-center gap-1.5">
                        {m.display_name}
                        {isHighlight && (
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-3 font-sans">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                        isAdaptive 
                          ? 'bg-indigo-950 text-indigo-300 border border-indigo-800'
                          : 'bg-slate-800 text-slate-300 border border-slate-700'
                      }`}>
                        {isAdaptive ? 'Streaming' : 'Held-Out'}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-slate-400">{m.threshold.toFixed(2)}</td>
                    <td className="py-3 px-3 text-right font-semibold text-cyan-400">
                      {(m.recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right text-slate-200">
                      {(m.precision * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-bold text-emerald-400">
                      {m.f1.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right text-indigo-300">
                      {m.pr_auc.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-300">
                      {m.roc_auc.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right text-rose-400 font-semibold">{m.fp}</td>
                    <td className="py-3 px-3 text-right text-amber-400">{m.fn}</td>
                    <td className="py-3 px-3 text-right text-slate-400 font-sans">
                      {(m.latency_ms || 0.035).toFixed(3)} ms
                    </td>
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

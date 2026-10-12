import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  ChevronLeft, 
  ChevronRight, 
  ShieldAlert, 
  CheckCircle2, 
  AlertTriangle,
  Eye,
  EyeOff
} from 'lucide-react';
import { fetchTransactions } from '../api';

export default function TransactionExplorerTab({ onSelectTransaction }) {
  const [strategy, setStrategy] = useState('guided_fuzzy_frozen');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [viewMode, setViewMode] = useState('offline_research_evaluation'); // offline_research_evaluation vs operational_simulation
  const [filterPred, setFilterPred] = useState('all');
  const [filterActual, setFilterActual] = useState('all');
  const [searchId, setSearchId] = useState('');
  const [minWidth, setMinWidth] = useState(0.0);

  const [loading, setLoading] = useState(false);
  const [txResponse, setTxResponse] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadTransactions();
  }, [strategy, page, pageSize, viewMode, filterPred, filterActual, minWidth]);

  async function loadTransactions() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchTransactions({
        strategy,
        page,
        pageSize,
        viewMode,
        filterPred,
        filterActual,
        minWidth,
        searchId,
      });
      setTxResponse(res);
    } catch (err) {
      setError(err.message || 'Failed to load transaction predictions');
    } finally {
      setLoading(false);
    }
  }

  function handleSearchSubmit(e) {
    e.preventDefault();
    setPage(1);
    loadTransactions();
  }

  return (
    <div className="space-y-6">
      {/* Header and Operational Mode Switcher */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Search className="w-5 h-5 text-cyan-400" />
            Transaction Stream Explorer
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Browse, filter, and inspect individual transactions across the 56,962 test stream
          </p>
        </div>

        {/* Operational vs Research View Toggle */}
        <div className="flex items-center gap-2 bg-slate-800/80 p-1 rounded-lg border border-slate-700">
          <button
            onClick={() => setViewMode('offline_research_evaluation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              viewMode === 'offline_research_evaluation'
                ? 'bg-cyan-600 text-white'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            Research Mode (Labels Visible)
          </button>
          <button
            onClick={() => setViewMode('operational_simulation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              viewMode === 'operational_simulation'
                ? 'bg-amber-600 text-white'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <EyeOff className="w-3.5 h-3.5" />
            Operational Mode (Labels Masked)
          </button>
        </div>
      </div>

      {viewMode === 'operational_simulation' && (
        <div className="p-3.5 rounded-lg bg-amber-950/40 border border-amber-800 text-amber-200 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          <span>
            <strong>Operational Simulation Active:</strong> Ground-truth labels are strictly masked to represent live point-in-time transaction arrival before any chargeback or customer feedback occurs.
          </span>
        </div>
      )}

      {/* Filter Toolbar */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-3">
        <form onSubmit={handleSearchSubmit} className="flex flex-wrap items-center gap-3">
          {/* Strategy selector */}
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-slate-400">Stream:</span>
            <select
              value={strategy}
              onChange={(e) => { setStrategy(e.target.value); setPage(1); }}
              className="bg-slate-950 border border-slate-700 text-slate-200 rounded px-2.5 py-1.5 font-medium text-xs focus:ring-1 focus:ring-cyan-500"
            >
              <option value="guided_fuzzy_frozen">Guided Fuzzy (Frozen)</option>
              <option value="guided_fuzzy_adaptive_delayed_1h">Guided Fuzzy (1-hr Delay)</option>
              <option value="guided_fuzzy_adaptive_immediate">Guided Fuzzy (Immediate)</option>
              <option value="logistic_regression_frozen">Logistic Regression (Frozen)</option>
              <option value="logistic_regression_adaptive_immediate">Logistic Regression (Adaptive)</option>
            </select>
          </div>

          {/* Search by ID */}
          <div className="flex items-center gap-1.5 text-xs">
            <input
              type="text"
              value={searchId}
              onChange={(e) => setSearchId(e.target.value)}
              placeholder="Search by Index ID..."
              className="bg-slate-950 border border-slate-700 text-slate-100 rounded px-3 py-1.5 w-44 text-xs font-mono focus:ring-1 focus:ring-cyan-500"
            />
            <button
              type="submit"
              className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-white text-xs font-medium"
            >
              Search
            </button>
          </div>

          {/* Filter predicted class */}
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-slate-400">Predicted:</span>
            <select
              value={filterPred}
              onChange={(e) => { setFilterPred(e.target.value); setPage(1); }}
              className="bg-slate-950 border border-slate-700 text-slate-200 rounded px-2.5 py-1.5 text-xs"
            >
              <option value="all">All Verdicts</option>
              <option value="fraud">Fraud Alerts</option>
              <option value="legitimate">Approved Legit</option>
            </select>
          </div>

          {/* Filter actual class (only in research mode) */}
          {viewMode === 'offline_research_evaluation' && (
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-slate-400">Actual Label:</span>
              <select
                value={filterActual}
                onChange={(e) => { setFilterActual(e.target.value); setPage(1); }}
                className="bg-slate-950 border border-slate-700 text-slate-200 rounded px-2.5 py-1.5 text-xs"
              >
                <option value="all">All Labels</option>
                <option value="fraud">Actual Frauds (1)</option>
                <option value="legitimate">Actual Legit (0)</option>
              </select>
            </div>
          )}

          {/* Page size */}
          <div className="flex items-center gap-1.5 text-xs ml-auto">
            <span className="text-slate-400">Rows:</span>
            <select
              value={pageSize}
              onChange={(e) => { setPageSize(parseInt(e.target.value, 10)); setPage(1); }}
              className="bg-slate-950 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs"
            >
              <option value="15">15</option>
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
          </div>
        </form>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/50 border border-rose-800 text-rose-200 text-xs">
          {error}
        </div>
      )}

      {/* Transaction Table */}
      <div className="rounded-xl bg-slate-900 border border-slate-800 shadow-md overflow-hidden">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <div>
            Showing {txResponse ? txResponse.data.length : 0} of {txResponse ? txResponse.total_matched.toLocaleString() : 0} matched transactions
          </div>
          {txResponse && (
            <div>Page {txResponse.page} of {txResponse.total_pages}</div>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-slate-300">
            <thead className="text-[11px] uppercase tracking-wider bg-slate-950/60 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3 px-4 font-semibold">Tx ID</th>
                <th className="py-3 px-3 font-semibold text-right">Time (s)</th>
                <th className="py-3 px-3 font-semibold text-right">Expected Prob</th>
                <th className="py-3 px-3 font-semibold text-center">Fuzzy Bounds [p_left, p_right]</th>
                <th className="py-3 px-3 font-semibold text-right">Fuzzy Width</th>
                <th className="py-3 px-3 font-semibold text-center">Predicted Verdict</th>
                {viewMode === 'offline_research_evaluation' && (
                  <th className="py-3 px-3 font-semibold text-center">True Label</th>
                )}
                <th className="py-3 px-3 font-semibold text-center">Review Triage</th>
                <th className="py-3 px-3 font-semibold text-center">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono text-[11px]">
              {loading ? (
                <tr>
                  <td colSpan={viewMode === 'offline_research_evaluation' ? 9 : 8} className="py-12 text-center text-slate-400 font-sans">
                    Loading transactions...
                  </td>
                </tr>
              ) : txResponse && txResponse.data.length > 0 ? (
                txResponse.data.map((tx) => {
                  const isFraudPred = tx.predicted_class === 1;
                  const isFraudActual = tx.true_label === 1;
                  const isCorrect = isFraudPred === isFraudActual;

                  return (
                    <tr key={tx.index} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-2.5 px-4 font-bold text-slate-200">#{tx.index}</td>
                      <td className="py-2.5 px-3 text-right text-slate-400">{tx.timestamp.toFixed(1)}</td>
                      <td className="py-2.5 px-3 text-right font-bold text-cyan-400">
                        {(tx.expected_value * 100).toFixed(2)}%
                      </td>
                      <td className="py-2.5 px-3 text-center text-slate-300">
                        [{(tx.p_left * 100).toFixed(1)}%, {(tx.p_right * 100).toFixed(1)}%]
                      </td>
                      <td className="py-2.5 px-3 text-right text-indigo-300">
                        {(tx.uncertainty_width * 100).toFixed(2)}%
                      </td>
                      <td className="py-2.5 px-3 text-center font-sans">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          isFraudPred 
                            ? 'bg-rose-950 text-rose-300 border border-rose-800' 
                            : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                        }`}>
                          {isFraudPred ? 'FRAUD' : 'LEGIT'}
                        </span>
                      </td>
                      {viewMode === 'offline_research_evaluation' && (
                        <td className="py-2.5 px-3 text-center font-sans">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            isFraudActual
                              ? 'bg-rose-950/80 text-rose-200'
                              : 'bg-slate-800 text-slate-300'
                          }`}>
                            {isFraudActual ? 'Fraud (1)' : 'Legit (0)'}
                          </span>
                        </td>
                      )}
                      <td className="py-2.5 px-3 text-center font-sans">
                        {tx.routed_to_review_margin ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-amber-950 text-amber-300 border border-amber-800">
                            Queue
                          </span>
                        ) : (
                          <span className="text-[10px] text-slate-500">Auto</span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-center font-sans">
                        <button
                          onClick={() => onSelectTransaction(tx.index)}
                          className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold"
                        >
                          Explain
                        </button>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={viewMode === 'offline_research_evaluation' ? 9 : 8} className="py-12 text-center text-slate-400 font-sans">
                    No transactions match the selected filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        {txResponse && txResponse.total_pages > 1 && (
          <div className="p-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-300">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="flex items-center gap-1 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:hover:bg-slate-800 font-medium transition-colors"
            >
              <ChevronLeft className="w-4 h-4" /> Previous
            </button>
            <span>
              Page <strong>{page}</strong> of <strong>{txResponse.total_pages}</strong>
            </span>
            <button
              onClick={() => setPage(p => Math.min(txResponse.total_pages, p + 1))}
              disabled={page >= txResponse.total_pages}
              className="flex items-center gap-1 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:hover:bg-slate-800 font-medium transition-colors"
            >
              Next <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

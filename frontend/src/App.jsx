import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import OverviewTab from './components/OverviewTab';
import ModelComparisonTab from './components/ModelComparisonTab';
import StreamingAdaptationTab from './components/StreamingAdaptationTab';
import FuzzyExplanationTab from './components/FuzzyExplanationTab';
import TriageTab from './components/TriageTab';
import TransactionExplorerTab from './components/TransactionExplorerTab';
import MethodologyTab from './components/MethodologyTab';

import { 
  fetchHealth, 
  fetchSummary, 
  fetchModelComparison, 
  fetchStreamingWindows, 
  fetchTriageResults 
} from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [healthStatus, setHealthStatus] = useState(null);
  const [summary, setSummary] = useState(null);
  const [models, setModels] = useState(null);
  const [windows, setWindows] = useState(null);
  const [triage, setTriage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Inspector transaction link state
  const [selectedTxId, setSelectedTxId] = useState('541');

  useEffect(() => {
    loadDashboardData();
  }, []);

  async function loadDashboardData() {
    setLoading(true);
    setError(null);
    try {
      const [h, s, m, w, t] = await Promise.all([
        fetchHealth().catch(() => ({ status: 'offline' })),
        fetchSummary(),
        fetchModelComparison(),
        fetchStreamingWindows(),
        fetchTriageResults(),
      ]);
      setHealthStatus(h);
      setSummary(s);
      setModels(m);
      setWindows(w);
      setTriage(t);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      setError(err.message || 'Failed to connect to FuzzyGuard AI backend.');
    } finally {
      setLoading(false);
    }
  }

  function handleSelectTransaction(id) {
    setSelectedTxId(String(id));
    setActiveTab('explanation');
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      <Header 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        healthStatus={healthStatus} 
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-200 text-xs flex items-center justify-between">
            <div>
              <strong>Connection Error:</strong> {error}
            </div>
            <button
              onClick={loadDashboardData}
              className="px-3 py-1 rounded bg-rose-800 hover:bg-rose-700 text-white font-medium text-xs transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {loading ? (
          <div className="py-32 text-center text-slate-400 space-y-3">
            <div className="inline-block animate-spin w-10 h-10 border-3 border-cyan-500 border-t-transparent rounded-full"></div>
            <p className="text-sm font-medium">Initializing FuzzyGuard AI Research Environment...</p>
            <p className="text-xs text-slate-500">Loading model weights, prequential predictions, and triage metrics</p>
          </div>
        ) : (
          <div>
            {activeTab === 'overview' && (
              <OverviewTab 
                summary={summary} 
                models={models} 
                onSelectTab={setActiveTab} 
              />
            )}

            {activeTab === 'comparison' && (
              <ModelComparisonTab 
                models={models} 
              />
            )}

            {activeTab === 'streaming' && (
              <StreamingAdaptationTab 
                windows={windows} 
              />
            )}

            {activeTab === 'explanation' && (
              <FuzzyExplanationTab 
                initialTxId={selectedTxId} 
              />
            )}

            {activeTab === 'triage' && (
              <TriageTab 
                triage={triage} 
              />
            )}

            {activeTab === 'explorer' && (
              <TransactionExplorerTab 
                onSelectTransaction={handleSelectTransaction} 
              />
            )}

            {activeTab === 'methodology' && (
              <MethodologyTab />
            )}
          </div>
        )}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-400">FuzzyGuard AI</span>
            <span>&bull;</span>
            <span>Version 1.0.0 (Milestone 5)</span>
          </div>
          <div>
            Based on Charizanos, Demirhan, & İçen (2024) replication & guided learning extensions
          </div>
          <div className="text-[11px] text-slate-600">
            For academic research & evaluation benchmarking only
          </div>
        </div>
      </footer>
    </div>
  );
}

import React from 'react';
import { 
  ShieldCheck, 
  Activity, 
  BarChart3, 
  Layers, 
  Search, 
  Users, 
  BookOpen, 
  AlertTriangle 
} from 'lucide-react';

export default function Header({ activeTab, setActiveTab, healthStatus }) {
  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'comparison', label: 'Model Comparison', icon: BarChart3 },
    { id: 'streaming', label: 'Streaming Adaptation', icon: Layers },
    { id: 'explanation', label: 'Fuzzy Explanation', icon: ShieldCheck },
    { id: 'triage', label: 'Human-Review Triage', icon: Users },
    { id: 'explorer', label: 'Transaction Explorer', icon: Search },
    { id: 'methodology', label: 'Methodology & Notes', icon: BookOpen },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      {/* Simulation Disclaimer Banner */}
      <div className="bg-amber-950/70 border-b border-amber-800/60 px-4 py-1.5 text-xs text-amber-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span>
            <strong className="font-semibold">Research Prototype Notice:</strong> Historical chronological evaluation simulation using European cardholder transactions (2013). This is an offline research benchmark, not a live production banking deployment.
          </span>
        </div>
        <div className="hidden md:flex items-center gap-2 text-slate-400">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-[11px] text-emerald-400 font-mono">Backend Connected</span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-tr from-cyan-600 to-indigo-600 text-white shadow-lg shadow-cyan-500/20">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">FuzzyGuard AI</h1>
                <span className="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider bg-cyan-950 text-cyan-300 border border-cyan-800 uppercase">
                  Research Dashboard
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Online Fuzzy Credit Card Fraud Detection (Charizanos et al. Replication & Adaptation)
              </p>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex space-x-1 overflow-x-auto pb-1 text-sm no-scrollbar">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-2.5 rounded-t-lg font-medium whitespace-nowrap transition-colors border-b-2 ${
                  isActive
                    ? 'border-cyan-500 text-cyan-400 bg-slate-800/70 shadow-sm'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

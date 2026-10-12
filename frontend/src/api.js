// API Client for FuzzyGuard AI Research Dashboard
const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Failed to fetch health check');
  return res.json();
}

export async function fetchSummary() {
  const res = await fetch(`${API_BASE}/summary`);
  if (!res.ok) throw new Error('Failed to fetch summary');
  return res.json();
}

export async function fetchModelComparison(evaluationType = 'all') {
  const res = await fetch(`${API_BASE}/models/comparison?evaluation_type=${evaluationType}`);
  if (!res.ok) throw new Error('Failed to fetch model comparison');
  return res.json();
}

export async function fetchStreamingWindows(strategy = 'all') {
  const res = await fetch(`${API_BASE}/streaming/windows?strategy=${strategy}`);
  if (!res.ok) throw new Error('Failed to fetch streaming windows');
  return res.json();
}

export async function fetchTriageResults(protocol = 'all') {
  const res = await fetch(`${API_BASE}/streaming/triage?protocol=${protocol}`);
  if (!res.ok) throw new Error('Failed to fetch triage results');
  return res.json();
}

export async function fetchTransactions({
  strategy = 'guided_fuzzy_frozen',
  page = 1,
  pageSize = 25,
  viewMode = 'offline_research_evaluation',
  filterPred = 'all',
  filterActual = 'all',
  minProb = 0.0,
  maxProb = 1.0,
  minWidth = 0.0,
  searchId = '',
}) {
  const params = new URLSearchParams({
    strategy,
    page: String(page),
    page_size: String(pageSize),
    view_mode: viewMode,
    filter_pred: filterPred,
    filter_actual: filterActual,
    min_prob: String(minProb),
    max_prob: String(maxProb),
    min_width: String(minWidth),
  });
  if (searchId) {
    params.append('search_id', searchId);
  }
  const res = await fetch(`${API_BASE}/transactions?${params.toString()}`);
  if (!res.ok) throw new Error('Failed to fetch transactions');
  return res.json();
}

export async function fetchTransactionExplanation(id) {
  const res = await fetch(`${API_BASE}/transactions/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch explanation for transaction ${id}`);
  return res.json();
}

export async function predictCustomTransaction(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Failed to score custom transaction');
  return res.json();
}

export async function fetchCoefficients() {
  const res = await fetch(`${API_BASE}/coefficients`);
  if (!res.ok) throw new Error('Failed to fetch coefficients');
  return res.json();
}

export async function fetchMethodology() {
  const res = await fetch(`${API_BASE}/methodology`);
  if (!res.ok) throw new Error('Failed to fetch methodology');
  return res.json();
}

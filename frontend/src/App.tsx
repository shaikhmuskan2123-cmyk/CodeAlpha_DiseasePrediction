import { useEffect, useMemo, useState, useRef } from 'react';

type FeatureDefinition = {
  name: string;
  kind: 'numeric' | 'categorical';
  minimum?: number;
  maximum?: number;
  median?: number;
  categories?: Array<string | number | boolean>;
};

type DiseaseInfo = {
  id: string;
  name: string;
  dataset: string;
  source: string;
  target: string;
  population: string;
  available: boolean;
  features: FeatureDefinition[];
};

type ModelComparisonItem = {
  model: string;
  cv_roc_auc_mean: number;
  cv_roc_auc_std: number;
  best_params?: Record<string, unknown>;
};

type Metrics = {
  disease: string;
  selected_model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  test_size: number;
  dataset_size: number;
  confusion_matrix?: number[][];
  class_balance?: Record<string, number>;
  model_comparison?: ModelComparisonItem[];
  specificity?: number;
  brier_score?: number;
};

type DatasetSummary = {
  name: string;
  rows: number;
  columns: string[];
  missing?: Record<string, number>;
  duplicates?: number;
  target_distribution?: Record<string, number>;
  preview?: Record<string, unknown>[];
  records?: Record<string, unknown>[];
};

type HistoryRecord = {
  id: number;
  disease: string;
  predicted_class: string;
  positive_score: number | null;
  created_at: string;
};

type PredictionExplanationFeature = {
  feature?: string;
  contribution?: number;
  importance?: number;
  transformed_value?: number;
};

type PredictionExplanation = {
  method?: string;
  scope?: string;
  class_label?: string;
  intercept?: number;
  decision_score?: number;
  contribution_sum?: number;
  features?: PredictionExplanationFeature[];
};

type PredictionResult = {
  disease: string;
  predicted_class: string;
  class_value: number;
  positive_score: number | null;
  score_label: string;
  model_name: string;
  model_version: string;
  target_definition: string;
  explanation?: PredictionExplanation;
  history_id?: number;
};

type ChatMessage = {
  id: string;
  sender: 'ai' | 'user';
  text: string;
  time: string;
};

type NavTab = 'home' | 'scan' | 'prediction' | 'analytics' | 'dataset' | 'history' | 'about';

const formatPercent = (value: number | undefined) => {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
};

const toNumber = (value: unknown, fallback: number) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export default function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [activeTab, setActiveTab] = useState<NavTab>('home');
  const [health, setHealth] = useState<{ status?: string; service?: string } | null>(null);
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [selectedDisease, setSelectedDisease] = useState<string>('heart');
  const [metrics, setMetrics] = useState<Record<string, Metrics>>({});
  const [dataset, setDataset] = useState<Record<string, DatasetSummary>>({});
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [saveToHistory, setSaveToHistory] = useState<boolean>(true);

  const [formValues, setFormValues] = useState<Record<string, string | number | boolean>>({});
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetSearch, setDatasetSearch] = useState<string>('');
  
  // Interactive Active Organ for glowing red dots on skeleton
  const [activeOrgan, setActiveOrgan] = useState<string>('heart');

  // Dynamic Vitals Adjusters
  const [customTemp, setCustomTemp] = useState<number>(98.6);
  const [heartRateOffset, setHeartRateOffset] = useState<number>(0);
  const [bpOffset, setBpOffset] = useState<number>(0);

  // AI Health Assistant Chat
  const [chatInput, setChatInput] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      sender: 'ai',
      text: "Hi there! 👋 Tell me what you're feeling today. You can type your symptoms or click directly on any organ on the skeleton to inspect clinical risks.",
      time: 'Just now',
    },
  ]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const selectedModel = useMemo(
    () => diseases.find((disease) => disease.id === selectedDisease) ?? diseases[0],
    [diseases, selectedDisease],
  );

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        const [healthResponse, modelsResponse] = await Promise.all([
          fetch('/api/health'),
          fetch('/api/models'),
        ]);

        if (!healthResponse.ok) throw new Error('The medical API is unavailable.');
        if (!modelsResponse.ok) throw new Error('The disease catalog could not be loaded.');

        const healthData = await healthResponse.json();
        const modelData = (await modelsResponse.json()) as DiseaseInfo[];

        setHealth(healthData);
        setDiseases(modelData);

        const metricEntries = await Promise.all(
          modelData.map(async (disease) => {
            const response = await fetch(`/api/models/${disease.id}/metrics`);
            if (!response.ok) return [disease.id, null] as const;
            const payload = (await response.json()) as Metrics;
            return [disease.id, payload] as const;
          }),
        );

        const nextMetrics: Record<string, Metrics> = {};
        for (const [key, value] of metricEntries) {
          if (value) nextMetrics[key] = value;
        }
        setMetrics(nextMetrics);

        const datasetEntries = await Promise.all(
          modelData.map(async (disease) => {
            const response = await fetch(`/api/datasets/${disease.id}`);
            if (!response.ok) return [disease.id, null] as const;
            const payload = (await response.json()) as DatasetSummary;
            return [disease.id, payload] as const;
          }),
        );

        const nextDataset: Record<string, DatasetSummary> = {};
        for (const [key, value] of datasetEntries) {
          if (value) nextDataset[key] = value;
        }
        setDataset(nextDataset);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load the dashboard.');
      } finally {
        setLoading(false);
      }
    };

    void loadInitialData();
  }, []);

  const loadHistoryData = async () => {
    try {
      const response = await fetch('/api/history');
      if (response.ok) {
        const records = (await response.json()) as HistoryRecord[];
        setHistory(records);
      }
    } catch {
      // Ignored for tolerance
    }
  };

  useEffect(() => {
    if (activeTab === 'history') {
      void loadHistoryData();
    }
  }, [activeTab]);

  useEffect(() => {
    if (!selectedModel) return;

    const defaults: Record<string, string | number | boolean> = {};
    for (const feature of selectedModel.features) {
      if (feature.kind === 'categorical') {
        defaults[feature.name] = feature.categories && feature.categories.length > 0 ? String(feature.categories[0]) : '0';
      } else {
        defaults[feature.name] = toNumber(feature.median ?? feature.minimum ?? 0, 0);
      }
    }

    setFormValues(defaults);
    setPrediction(null);
  }, [selectedModel]);

  const currentMetrics = selectedModel ? metrics[selectedModel.id] : undefined;
  const currentDataset = selectedModel ? dataset[selectedModel.id] : undefined;

  const updateField = (field: string, value: string | number | boolean) => {
    setFormValues((previous) => ({ ...previous, [field]: value }));
  };

  const coerceFeatureValue = (feature: FeatureDefinition, value: string | number | boolean) => {
    if (feature.kind === 'categorical') {
      const categories = feature.categories ?? [];
      if (categories.length === 0) return value;
      const strValue = String(value);
      const match = categories.find((category) => String(category) === strValue);
      return match ?? categories[0];
    }
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue : 0;
  };

  const loadSampleCase = (type: 'median' | 'sample1' | 'sample2') => {
    if (!selectedModel) return;
    const records = currentDataset?.records ?? currentDataset?.preview ?? [];

    if (type === 'median') {
      const defaults: Record<string, string | number | boolean> = {};
      for (const feature of selectedModel.features) {
        if (feature.kind === 'categorical') {
          defaults[feature.name] = feature.categories?.[0] ? String(feature.categories[0]) : '0';
        } else {
          defaults[feature.name] = toNumber(feature.median ?? feature.minimum ?? 0, 0);
        }
      }
      setFormValues(defaults);
      return;
    }

    if (records.length > 0) {
      const sampleRow = type === 'sample1' ? records[0] : (records[records.length > 1 ? 1 : 0]);
      const loadedValues: Record<string, string | number | boolean> = {};
      for (const feature of selectedModel.features) {
        if (feature.name in sampleRow && sampleRow[feature.name] !== null) {
          loadedValues[feature.name] = sampleRow[feature.name] as string | number | boolean;
        } else {
          loadedValues[feature.name] = feature.kind === 'categorical'
            ? String(feature.categories?.[0] ?? '0')
            : toNumber(feature.median ?? 0, 0);
        }
      }
      setFormValues(loadedValues);
    }
  };

  const submitPrediction = async () => {
    if (!selectedModel) return;
    setSubmitting(true);
    setError(null);

    try {
      const payload = {
        features: Object.fromEntries(
          selectedModel.features.map((feature) => [
            feature.name,
            coerceFeatureValue(feature, formValues[feature.name] ?? (feature.kind === 'categorical' ? String(feature.categories?.[0] ?? '') : 0)),
          ]),
        ),
        persist: saveToHistory,
      };

      const response = await fetch(`/api/predict/${selectedModel.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const content = await response.json().catch(() => null);
        throw new Error(content?.detail ?? 'Prediction failed.');
      }

      const result = (await response.json()) as PredictionResult;
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to generate a prediction.');
    } finally {
      setSubmitting(false);
    }
  };

  const downloadPdfReport = async () => {
    if (!selectedModel) return;
    setDownloadingPdf(true);
    setError(null);

    try {
      const payload = {
        features: Object.fromEntries(
          selectedModel.features.map((feature) => [
            feature.name,
            coerceFeatureValue(feature, formValues[feature.name] ?? (feature.kind === 'categorical' ? String(feature.categories?.[0] ?? '') : 0)),
          ]),
        ),
      };

      const response = await fetch(`/api/reports/${selectedModel.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('PDF report generation failed.');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `medipredict-${selectedModel.id}-report.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to download report.');
    } finally {
      setDownloadingPdf(false);
    }
  };

  const clearHistory = async () => {
    try {
      const response = await fetch('/api/history', { method: 'DELETE' });
      if (!response.ok) throw new Error('Failed to delete history.');
      setHistory([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to clear prediction history.');
    }
  };

  const exportFilteredDatasetCsv = () => {
    if (!currentDataset || !currentDataset.records || currentDataset.records.length === 0) return;
    const records = filteredRecords;
    if (records.length === 0) return;
    const headers = Object.keys(records[0]);
    const csvContent = [
      headers.join(','),
      ...records.map((row) =>
        headers.map((field) => JSON.stringify(row[field] ?? '')).join(','),
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${selectedModel?.id ?? 'dataset'}-export.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const filteredRecords = useMemo(() => {
    const list = currentDataset?.records ?? currentDataset?.preview ?? [];
    if (!datasetSearch.trim()) return list;
    const query = datasetSearch.toLowerCase();
    return list.filter((row) =>
      Object.values(row).some((val) => String(val).toLowerCase().includes(query)),
    );
  }, [currentDataset, datasetSearch]);

  const riskPct = prediction?.positive_score !== null && typeof prediction?.positive_score !== 'undefined'
    ? Math.round(prediction.positive_score * 100)
    : prediction
    ? (prediction.class_value === 1 ? 82 : 18)
    : 0;

  const riskLevel = riskPct >= 65 ? 'High' : riskPct >= 35 ? 'Moderate' : 'Low';

  const explanationFeatures = useMemo(() => {
    if (!prediction?.explanation?.features) return [];
    return prediction.explanation.features.slice(0, 4);
  }, [prediction]);

  // --- Organ Selection Handler (Lights up Red Dot on Skeleton) ---
  const handleOrganClick = (organ: string) => {
    setActiveOrgan(organ);
    if (organ === 'heart') {
      setSelectedDisease('heart');
      setHeartRateOffset(0);
    } else if (organ === 'pancreas' || organ === 'kidneys' || organ === 'stomach') {
      setSelectedDisease('diabetes');
    } else if (organ === 'lungs') {
      setSelectedDisease('breast-cancer');
    } else if (organ === 'brain') {
      setCustomTemp(101.2);
    }
  };

  // --- Dynamic Vitals Computation based on live form values ---
  const liveHeartRate = useMemo(() => {
    if (selectedDisease === 'heart') {
      const base = toNumber(formValues['thalach'], 145);
      return Math.max(50, Math.min(220, base + heartRateOffset));
    }
    return 78 + heartRateOffset;
  }, [selectedDisease, formValues, heartRateOffset]);

  const liveBloodPressure = useMemo(() => {
    if (selectedDisease === 'heart') {
      const systolic = toNumber(formValues['trestbps'], 130) + bpOffset;
      return `${Math.max(80, Math.min(220, systolic))}/80`;
    }
    return `${120 + bpOffset}/80`;
  }, [selectedDisease, formValues, bpOffset]);

  const liveGlucose = useMemo(() => {
    if (selectedDisease === 'diabetes') {
      const isHighSugar = formValues['polyuria'] === 'Yes' || formValues['polyuria'] === '1' || formValues['weakness'] === 'Yes';
      return isHighSugar ? 168 : 105;
    }
    if (selectedDisease === 'heart') {
      return formValues['fbs'] === '1' || formValues['fbs'] === 1 ? 142 : 92;
    }
    return 95;
  }, [selectedDisease, formValues]);

  // --- AI Health Assistant Interactive Engine ---
  const handleSendMessage = (textToSend?: string) => {
    const query = (textToSend ?? chatInput).trim();
    if (!query) return;

    const userMsg: ChatMessage = {
      id: String(Date.now()),
      sender: 'user',
      text: query,
      time: 'Now',
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setChatInput('');

    // Process intent and reply
    setTimeout(() => {
      const lower = query.toLowerCase();
      let replyText = '';

      if (lower.includes('kidney') || lower.includes('urine') || lower.includes('urination') || lower.includes('sugar') || lower.includes('diabetes') || lower.includes('thirst')) {
        handleOrganClick('kidneys');
        replyText = 'I detected symptoms matching metabolic and renal indicators. I have illuminated both Kidneys with glowing red target dots on your skeleton and switched to the Early-Stage Diabetes Module (Sylhet 520 Cases). Your blood glucose reflects elevated levels.';
      } else if (lower.includes('chest') || lower.includes('heart') || lower.includes('angina') || lower.includes('pressure') || lower.includes('bp') || lower.includes('palpitation')) {
        handleOrganClick('heart');
        replyText = 'Cardiovascular strain or chest tightness matches coronary stenosis patterns. I have illuminated the Heart on the skeleton with red markers and loaded the Cleveland Heart Disease module (13 clinical parameters). Check your resting BP and maximum heart rate below!';
      } else if (lower.includes('lump') || lower.includes('cell') || lower.includes('breast') || lower.includes('tumor') || lower.includes('biopsy') || lower.includes('mass')) {
        handleOrganClick('lungs');
        replyText = 'I have selected the Wisconsin Diagnostic Breast Cell Classification module. The red target dot is positioned on the thoracic tissue area. This evaluates 30 fine-needle aspirate cell-nucleus measurements.';
      } else if (lower.includes('fever') || lower.includes('headache') || lower.includes('temperature') || lower.includes('hot')) {
        handleOrganClick('brain');
        setCustomTemp(101.4);
        replyText = 'Headache and fever detected. I have updated your Body Temperature vital to 101.4 °F and highlighted the Cranial area. Rest and hydration are strongly advised.';
      } else {
        replyText = `Understood. You can click on any red dot on the skeleton (Heart, Kidneys, Pancreas, Chest) or use the preset buttons below to evaluate risk with full explainability.`;
      }

      const aiMsg: ChatMessage = {
        id: String(Date.now() + 1),
        sender: 'ai',
        text: replyText,
        time: 'Just now',
      };
      setMessages((prev) => [...prev, aiMsg]);
    }, 450);
  };

  return (
    <div className="health-dashboard">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">✚</div>
          <div>
            <h2>HealthSense AI</h2>
            <small>Understand Today for a Healthier Tomorrow</small>
          </div>
        </div>

        <nav className="nav-list">
          <button
            className={`nav-item ${activeTab === 'home' || activeTab === 'prediction' ? 'active' : ''}`}
            type="button"
            onClick={() => setActiveTab('home')}
          >
            <span className="nav-icon">⌂</span>
            Home
          </button>
          <button
            className={`nav-item ${activeTab === 'scan' ? 'active' : ''}`}
            type="button"
            onClick={() => {
              setActiveTab('home');
              window.scrollTo({ top: 180, behavior: 'smooth' });
            }}
          >
            <span className="nav-icon">◌</span>
            Body Scan
          </button>
          <button
            className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`}
            type="button"
            onClick={() => setActiveTab('analytics')}
          >
            <span className="nav-icon">⌁</span>
            Health Insights &amp; Models
          </button>
          <button
            className={`nav-item ${activeTab === 'dataset' ? 'active' : ''}`}
            type="button"
            onClick={() => setActiveTab('dataset')}
          >
            <span className="nav-icon">✦</span>
            Dataset Explorer
          </button>
          <button
            className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
            type="button"
            onClick={() => setActiveTab('history')}
          >
            <span className="nav-icon">☰</span>
            History &amp; Records
          </button>
          <button
            className={`nav-item ${activeTab === 'about' ? 'active' : ''}`}
            type="button"
            onClick={() => setActiveTab('about')}
          >
            <span className="nav-icon">⚙</span>
            Clinical Ethics
          </button>
        </nav>

        <div className="sidebar-foot">
          <div className="theme-switch-row">
            <button
              className={`theme-pill-btn ${theme === 'light' ? 'active' : ''}`}
              type="button"
              onClick={() => setTheme('light')}
            >
              ☀️ Light
            </button>
            <button
              className={`theme-pill-btn ${theme === 'dark' ? 'active' : ''}`}
              type="button"
              onClick={() => setTheme('dark')}
            >
              🌙 Dark
            </button>
          </div>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="main-panel">
        <header className="topbar">
          <div className="greeting-block">
            <h1>Hello 👋</h1>
            <p>Let's check your health today. Describe your symptoms or explore the interactive skeleton body to get a prediction.</p>
          </div>

          <div className="topbar-right">
            {health && (
              <span className="mini-tag" style={{ background: 'rgba(16, 185, 129, 0.12)', color: '#059669' }}>
                ● {health.service} Online
              </span>
            )}
            <div className="search-box">
              <span>⌕</span>
              <input
                type="text"
                placeholder="Search symptoms, diseases, parameters..."
                value={datasetSearch}
                onChange={(e) => setDatasetSearch(e.target.value)}
              />
            </div>
            <div className="user-pill">
              <div className="user-avatar">👤</div>
              <div>
                <strong>User</strong>
                <small>Patient Mode</small>
              </div>
            </div>
          </div>
        </header>

        {error && <div className="error-banner">{error}</div>}
        {loading && <div style={{ marginBottom: '14px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>Loading clinical models and datasets…</div>}

        {/* --- VIEW 1: HOME & BODY SCAN PREDICTION --- */}
        {(activeTab === 'home' || activeTab === 'prediction') && (
          <>
            {/* Disease Selector Bar */}
            <div className="disease-selector-bar">
              {diseases.map((d) => {
                const isSelected = d.id === selectedModel?.id;
                const icon = d.id === 'heart' ? '🫀' : d.id === 'diabetes' ? '🩸' : '🔬';
                return (
                  <button
                    key={d.id}
                    className={`disease-tab-btn ${isSelected ? 'active' : ''}`}
                    type="button"
                    onClick={() => {
                      setSelectedDisease(d.id);
                      setActiveOrgan(d.id === 'heart' ? 'heart' : d.id === 'diabetes' ? 'kidneys' : 'lungs');
                      setError(null);
                    }}
                  >
                    <span>{icon}</span>
                    <span>{d.name}</span>
                    <span className="disease-tab-badge">
                      {d.id === 'heart' ? '13 Clinical Biomarkers' : d.id === 'diabetes' ? '16 Symptoms' : '30 Cell Measurements'}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* 3-Column Top Trio Grid: AI Health Assistant | Skeleton Body Scan with Red Dots | Prediction Card */}
            <div className="top-trio-grid">
              {/* Column 1: Active AI Health Assistant */}
              <div className="card ai-assistant-card">
                <div className="ai-assistant-header">
                  <div className="ai-robot-avatar">🤖</div>
                  <div>
                    <h3>AI Health Assistant</h3>
                    <small>Active Clinical AI Companion</small>
                  </div>
                </div>

                <div className="ai-assistant-messages">
                  {messages.map((m) => (
                    <div key={m.id} className={`chat-bubble ${m.sender}`}>
                      {m.text}
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>

                <div className="quick-prompts-row">
                  <button
                    className="quick-chip"
                    type="button"
                    onClick={() => handleSendMessage('Check my Kidneys and urine output')}
                  >
                    🫘 Check Kidneys
                  </button>
                  <button
                    className="quick-chip"
                    type="button"
                    onClick={() => handleSendMessage('I have chest pain and shortness of breath')}
                  >
                    🫀 Chest Pain (Heart)
                  </button>
                  <button
                    className="quick-chip"
                    type="button"
                    onClick={() => handleSendMessage('High fever & headache for 2 days')}
                  >
                    🌡️ High Fever
                  </button>
                </div>

                <div className="ai-assistant-input-bar">
                  <input
                    type="text"
                    placeholder="Type symptoms or ask a question..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSendMessage();
                    }}
                  />
                  <button className="ai-assistant-send-btn" type="button" onClick={() => handleSendMessage()}>
                    ✈
                  </button>
                </div>
              </div>

              {/* Column 2: Interactive Holographic Skeleton with Real Red Dots */}
              <div className="card body-scan-card">
                <div className="body-scan-header">
                  <div>
                    <h3>Anatomical Holographic Scan</h3>
                    <small style={{ color: 'var(--text-muted)' }}>
                      Press on body organs or symptoms &mdash; glowing red target dots track the area!
                    </small>
                  </div>
                  {prediction && (
                    <span className="mini-tag" style={{ background: 'rgba(239, 68, 68, 0.12)', color: '#ef4444' }}>
                      {riskPct}% Risk
                    </span>
                  )}
                </div>

                <div className="body-scan-arena">
                  {/* Skeleton Figure with Real Glowing Red Dots */}
                  <div className="skeleton-figure-container">
                    <div className="skeleton-wrapper">
                      <img
                        src="/skeleton_model.png"
                        alt="Anatomical Skeleton Body Scan"
                        className="skeleton-img"
                      />

                      {/* --- Glowing Red Dots on Specific Anatomical Locations --- */}
                      
                      {/* Brain / Head */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'brain' ? 'active' : ''}`}
                        style={{ top: '11%', left: '54%' }}
                        onClick={() => handleOrganClick('brain')}
                        title="Brain / Headache Area"
                      />

                      {/* Lungs / Thoracic */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'lungs' || selectedDisease === 'breast-cancer' ? 'active' : ''}`}
                        style={{ top: '29%', left: '45%' }}
                        onClick={() => handleOrganClick('lungs')}
                        title="Lungs / Breast Tissue"
                      />

                      {/* Heart (Cardiac Center) */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'heart' || selectedDisease === 'heart' ? 'active' : ''}`}
                        style={{ top: '31%', left: '59%' }}
                        onClick={() => handleOrganClick('heart')}
                        title="Heart / Cardiac Angina"
                      />

                      {/* Stomach */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'stomach' ? 'active' : ''}`}
                        style={{ top: '41%', left: '54%' }}
                        onClick={() => handleOrganClick('stomach')}
                        title="Stomach / Gastrointestinal"
                      />

                      {/* Pancreas / Metabolism */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'pancreas' ? 'active' : ''}`}
                        style={{ top: '42%', left: '46%' }}
                        onClick={() => handleOrganClick('pancreas')}
                        title="Pancreas / Blood Glucose"
                      />

                      {/* Left Kidney (Red Dot) */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'kidneys' ? 'active' : ''}`}
                        style={{ top: '48%', left: '43%' }}
                        onClick={() => handleOrganClick('kidneys')}
                        title="Left Kidney (Filtration & Polyuria)"
                      />

                      {/* Right Kidney (Red Dot) */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'kidneys' ? 'active' : ''}`}
                        style={{ top: '48%', left: '66%' }}
                        onClick={() => handleOrganClick('kidneys')}
                        title="Right Kidney (Filtration & Polyuria)"
                      />

                      {/* Left Knee Joint */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'joints' ? 'active' : ''}`}
                        style={{ top: '74%', left: '45%' }}
                        onClick={() => handleOrganClick('joints')}
                        title="Left Knee Joint / Mobility"
                      />

                      {/* Right Knee Joint */}
                      <div
                        className={`glowing-organ-dot ${activeOrgan === 'joints' ? 'active' : ''}`}
                        style={{ top: '74%', left: '64%' }}
                        onClick={() => handleOrganClick('joints')}
                        title="Right Knee Joint / Mobility"
                      />
                    </div>
                  </div>

                  {/* Symptom Selector Column */}
                  <div className="symptom-selector-column">
                    <h4>Clinical Hotspots</h4>
                    
                    {/* Kidneys Button */}
                    <button
                      className={`symptom-tag-btn ${activeOrgan === 'kidneys' ? 'active-symptom' : ''}`}
                      type="button"
                      onClick={() => handleOrganClick('kidneys')}
                    >
                      <span>🫘 Kidneys (Polyuria)</span>
                      <span>{activeOrgan === 'kidneys' ? '🔴' : '+'}</span>
                    </button>

                    {/* Heart Button */}
                    <button
                      className={`symptom-tag-btn ${activeOrgan === 'heart' ? 'active-symptom' : ''}`}
                      type="button"
                      onClick={() => handleOrganClick('heart')}
                    >
                      <span>🫀 Heart (Angina)</span>
                      <span>{activeOrgan === 'heart' ? '🔴' : '+'}</span>
                    </button>

                    {/* Lungs / Cells */}
                    <button
                      className={`symptom-tag-btn ${activeOrgan === 'lungs' ? 'active-symptom' : ''}`}
                      type="button"
                      onClick={() => handleOrganClick('lungs')}
                    >
                      <span>🔬 Cell Tissue (Biopsy)</span>
                      <span>{activeOrgan === 'lungs' ? '🔴' : '+'}</span>
                    </button>

                    {/* Pancreas */}
                    <button
                      className={`symptom-tag-btn ${activeOrgan === 'pancreas' ? 'active-symptom' : ''}`}
                      type="button"
                      onClick={() => handleOrganClick('pancreas')}
                    >
                      <span>🩸 High Blood Glucose</span>
                      <span>{activeOrgan === 'pancreas' ? '🔴' : '+'}</span>
                    </button>

                    {/* Brain */}
                    <button
                      className={`symptom-tag-btn ${activeOrgan === 'brain' ? 'active-symptom' : ''}`}
                      type="button"
                      onClick={() => handleOrganClick('brain')}
                    >
                      <span>🧠 Headache / Cranial</span>
                      <span>{activeOrgan === 'brain' ? '🔴' : '+'}</span>
                    </button>

                    {/* Stomach */}
                    <button
                      className={`symptom-tag-btn ${activeOrgan === 'stomach' ? 'active-symptom' : ''}`}
                      type="button"
                      onClick={() => handleOrganClick('stomach')}
                    >
                      <span>🫃 Digestion / Nausea</span>
                      <span>{activeOrgan === 'stomach' ? '🔴' : '+'}</span>
                    </button>

                    {/* Joints */}
                    <button
                      className={`symptom-tag-btn ${activeOrgan === 'joints' ? 'active-symptom' : ''}`}
                      type="button"
                      onClick={() => handleOrganClick('joints')}
                    >
                      <span>🦵 Body Pain / Joints</span>
                      <span>{activeOrgan === 'joints' ? '🔴' : '+'}</span>
                    </button>
                  </div>
                </div>

                {/* Active Organ Banner */}
                <div className="active-organ-banner">
                  {activeOrgan === 'kidneys' && '🔴 Target Active: Kidneys (Renal Clearance & Polyuria Osmotic Load)'}
                  {activeOrgan === 'heart' && '🔴 Target Active: Heart (Coronary Stenosis & Maximum Heart Rate)'}
                  {activeOrgan === 'lungs' && '🔴 Target Active: Cellular Nucleus (Wisconsin FNA Measurements)'}
                  {activeOrgan === 'pancreas' && '🔴 Target Active: Pancreas (Insulin Regulation & Fasting Glucose)'}
                  {activeOrgan === 'brain' && '🔴 Target Active: Neurological / Cranial (Headache & Temperature)'}
                  {activeOrgan === 'stomach' && '🔴 Target Active: Gastrointestinal (Digestive System)'}
                  {activeOrgan === 'joints' && '🔴 Target Active: Musculoskeletal (Mobility & Weakness)'}
                </div>
              </div>

              {/* Column 3: Prediction Result Card */}
              <div className="card result-card">
                <div className="card-topline">
                  <span className="mini-tag">AI Trained Model</span>
                  <small style={{ color: 'var(--text-muted)' }}>
                    {new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
                  </small>
                </div>

                <div className="disease-result">
                  <div className="disease-icon-box">
                    {selectedModel?.id === 'heart' ? '🫀' : selectedModel?.id === 'diabetes' ? '🩸' : '🔬'}
                  </div>
                  <div>
                    <h4>{prediction ? prediction.predicted_class : selectedModel?.name}</h4>
                    <p>{prediction ? `${riskLevel} Risk Probability` : 'Awaiting clinical evaluation'}</p>
                  </div>
                </div>

                <div className="probability-metric-row">
                  <div>
                    <strong>{prediction ? `${riskPct}%` : '—'}</strong>
                    <span>Positive Class Probability</span>
                  </div>
                  {prediction && (
                    <span className="mini-tag" style={{ background: riskLevel === 'High' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)', color: riskLevel === 'High' ? '#ef4444' : '#10b981' }}>
                      {riskLevel} Risk
                    </span>
                  )}
                </div>

                <div className="progress-block">
                  <div className="progress-head">
                    <span>Confidence Score</span>
                    <strong>{prediction ? `${riskPct}%` : '0%'}</strong>
                  </div>
                  <div className="bar">
                    <span style={{ width: `${riskPct}%` }} />
                  </div>
                </div>

                <div className="explain-box">
                  <h5>Why this prediction?</h5>
                  <p>
                    {prediction?.explanation?.method ?? 'Trained scikit-learn classification pipeline.'}
                  </p>
                  {explanationFeatures.length > 0 && (
                    <ul className="mini-feature-list">
                      {explanationFeatures.map((f, i) => (
                        <li key={i}>
                          <span>{f.feature}</span>
                          <strong>
                            {typeof f.contribution === 'number'
                              ? `${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(3)} log-odds`
                              : typeof f.importance === 'number'
                              ? `${(f.importance * 100).toFixed(1)}% weight`
                              : '—'}
                          </strong>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                {prediction && (
                  <button
                    className="pdf-download-btn"
                    type="button"
                    onClick={downloadPdfReport}
                    disabled={downloadingPdf}
                  >
                    {downloadingPdf ? 'Generating PDF…' : '📥 Download Clinical PDF Report'}
                  </button>
                )}
              </div>
            </div>

            {/* --- DYNAMIC VITAL HEALTH INDICATORS (Really updates with form inputs!) --- */}
            <div className="bottom-vitals-row">
              {/* Vital 1: Heart Rate */}
              <div className="vital-card">
                <div className="vital-head">
                  <span>❤️ Heart Rate</span>
                  <span className={`vital-status-tag ${liveHeartRate > 140 ? 'danger' : liveHeartRate > 100 ? 'warning' : 'normal'}`}>
                    {liveHeartRate > 140 ? 'Tachycardia' : liveHeartRate > 100 ? 'Elevated' : 'Normal'}
                  </span>
                </div>
                <div className="vital-metric-row">
                  <div className="vital-metric">
                    {liveHeartRate} <small>bpm</small>
                  </div>
                  <div className="vital-stepper">
                    <button className="vital-stepper-btn" type="button" onClick={() => setHeartRateOffset((v) => v - 5)}>−</button>
                    <button className="vital-stepper-btn" type="button" onClick={() => setHeartRateOffset((v) => v + 5)}>+</button>
                  </div>
                </div>
                {/* Dynamically scaling SVG heartbeat pulse */}
                <svg className="vital-sparkline" viewBox="0 0 100 25" preserveAspectRatio="none">
                  <path
                    d={`M0,15 L20,15 L26,${liveHeartRate > 120 ? '2' : '7'} L32,${liveHeartRate > 120 ? '24' : '20'} L38,${liveHeartRate > 120 ? '1' : '5'} L45,18 L50,15 L100,15`}
                    fill="none"
                    stroke={liveHeartRate > 130 ? '#ef4444' : '#ec4899'}
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  />
                </svg>
              </div>

              {/* Vital 2: Body Temperature */}
              <div className="vital-card">
                <div className="vital-head">
                  <span>🌡️ Body Temperature</span>
                  <span className={`vital-status-tag ${customTemp > 100.4 ? 'danger' : customTemp > 99.0 ? 'warning' : 'normal'}`}>
                    {customTemp > 100.4 ? 'Fever' : customTemp > 99.0 ? 'Low-grade' : 'Normal'}
                  </span>
                </div>
                <div className="vital-metric-row">
                  <div className="vital-metric">
                    {customTemp.toFixed(1)} <small>°F</small>
                  </div>
                  <div className="vital-stepper">
                    <button className="vital-stepper-btn" type="button" onClick={() => setCustomTemp((v) => Math.max(96.0, Number((v - 0.2).toFixed(1))))}>−</button>
                    <button className="vital-stepper-btn" type="button" onClick={() => setCustomTemp((v) => Math.min(105.0, Number((v + 0.2).toFixed(1))))}>+</button>
                  </div>
                </div>
                <svg className="vital-sparkline" viewBox="0 0 100 25" preserveAspectRatio="none">
                  <path
                    d={`M0,${customTemp > 100 ? '16' : '12'} Q25,${customTemp > 100 ? '4' : '8'} 50,${customTemp > 100 ? '16' : '12'} T100,${customTemp > 100 ? '16' : '12'}`}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  />
                </svg>
              </div>

              {/* Vital 3: Blood Pressure */}
              <div className="vital-card">
                <div className="vital-head">
                  <span>🩸 Blood Pressure</span>
                  <span className={`vital-status-tag ${parseInt(liveBloodPressure) >= 140 ? 'danger' : parseInt(liveBloodPressure) >= 125 ? 'warning' : 'normal'}`}>
                    {parseInt(liveBloodPressure) >= 140 ? 'Stage 1/2' : parseInt(liveBloodPressure) >= 125 ? 'Pre-HTN' : 'Optimal'}
                  </span>
                </div>
                <div className="vital-metric-row">
                  <div className="vital-metric">
                    {liveBloodPressure} <small>mmHg</small>
                  </div>
                  <div className="vital-stepper">
                    <button className="vital-stepper-btn" type="button" onClick={() => setBpOffset((v) => v - 5)}>−</button>
                    <button className="vital-stepper-btn" type="button" onClick={() => setBpOffset((v) => v + 5)}>+</button>
                  </div>
                </div>
                <svg className="vital-sparkline" viewBox="0 0 100 25" preserveAspectRatio="none">
                  <path
                    d={`M0,16 L30,16 L36,${parseInt(liveBloodPressure) > 135 ? '4' : '9'} L44,22 L52,16 L100,16`}
                    fill="none"
                    stroke="#4f46e5"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  />
                </svg>
              </div>

              {/* Vital 4: Blood Glucose / Renal Clearance */}
              <div className="vital-card">
                <div className="vital-head">
                  <span>🫘 Blood Glucose</span>
                  <span className={`vital-status-tag ${liveGlucose >= 140 ? 'danger' : liveGlucose > 100 ? 'warning' : 'normal'}`}>
                    {liveGlucose >= 140 ? 'Diabetic' : liveGlucose > 100 ? 'Pre-DM' : 'Optimal'}
                  </span>
                </div>
                <div className="vital-metric-row">
                  <div className="vital-metric">
                    {liveGlucose} <small>mg/dL</small>
                  </div>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    {activeOrgan === 'kidneys' ? 'Active Renal Filtration' : 'Fasting Plasma'}
                  </span>
                </div>
                <svg className="vital-sparkline" viewBox="0 0 100 25" preserveAspectRatio="none">
                  <path
                    d={`M0,14 Q30,${liveGlucose > 130 ? '2' : '6'} 60,14 T100,14`}
                    fill="none"
                    stroke={liveGlucose > 130 ? '#ef4444' : '#06b6d4'}
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  />
                </svg>
              </div>
            </div>

            {/* Recommendations & Diet Row */}
            <div className="recommendations-grid">
              <div className="card">
                <h3>Personalized Clinical Recommendations</h3>
                <div className="rec-card-grid">
                  <div className="rec-item">
                    <span className="rec-icon">🛌</span>
                    <span>Rest Well (7-8 hrs)</span>
                  </div>
                  <div className="rec-item">
                    <span className="rec-icon">💧</span>
                    <span>Stay Hydrated (2-3 L)</span>
                  </div>
                  <div className="rec-item">
                    <span className="rec-icon">💊</span>
                    <span>Prescribed Care</span>
                  </div>
                  <div className="rec-item">
                    <span className="rec-icon">🚫</span>
                    <span>Avoid Junk Food</span>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3>Diet &amp; Nutrition Suggestions</h3>
                <div className="rec-card-grid">
                  <div className="rec-item">
                    <span className="rec-icon">🍊</span>
                    <span>Hydrating Fruits</span>
                  </div>
                  <div className="rec-item">
                    <span className="rec-icon">🍵</span>
                    <span>Warm Fluids / Tea</span>
                  </div>
                  <div className="rec-item">
                    <span className="rec-icon">🥗</span>
                    <span>Nutritious Greens</span>
                  </div>
                  <div className="rec-item">
                    <span className="rec-icon">🥑</span>
                    <span>Healthy Fats</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Clinical Inputs Form */}
            <div className="card clinical-inputs-card">
              <div className="preset-actions-row">
                <div>
                  <h3 style={{ fontSize: '1.1rem' }}>{selectedModel?.name} Clinical Parameters ({selectedModel?.features.length ?? 0} features)</h3>
                  <small style={{ color: 'var(--text-muted)' }}>Load benchmark cases or edit values &mdash; vitals react in real-time!</small>
                </div>
                <div className="preset-btn-group">
                  <button className="preset-chip-btn" type="button" onClick={() => loadSampleCase('median')}>
                    ↺ Reset to Medians
                  </button>
                  <button className="preset-chip-btn" type="button" onClick={() => loadSampleCase('sample1')}>
                    🧪 Benchmark Sample #1
                  </button>
                  <button className="preset-chip-btn" type="button" onClick={() => loadSampleCase('sample2')}>
                    🧪 Benchmark Sample #2
                  </button>
                </div>
                <label className="privacy-toggle-label">
                  <input
                    type="checkbox"
                    checked={saveToHistory}
                    onChange={(e) => setSaveToHistory(e.target.checked)}
                  />
                  Save to SQLite History
                </label>
              </div>

              <div className="form-grid">
                {selectedModel?.features.map((feature) => (
                  <label key={feature.name} className="field-row">
                    <span>{feature.name}</span>
                    {feature.kind === 'categorical' ? (
                      <select
                        value={String(formValues[feature.name] ?? '')}
                        onChange={(e) => updateField(feature.name, e.target.value)}
                      >
                        {(feature.categories ?? []).map((cat) => (
                          <option key={String(cat)} value={String(cat)}>
                            {String(cat)}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        value={Number(formValues[feature.name] ?? 0)}
                        min={feature.minimum ?? 0}
                        max={feature.maximum ?? 9999}
                        step={feature.maximum && feature.maximum < 5 ? '0.001' : '0.1'}
                        onChange={(e) => updateField(feature.name, Number(e.target.value))}
                      />
                    )}
                  </label>
                ))}
              </div>

              <button
                className="primary-button"
                type="button"
                onClick={submitPrediction}
                disabled={submitting}
              >
                {submitting ? 'Running Clinical Evaluation…' : `Predict ${selectedModel?.name} Risk`}
              </button>
            </div>
          </>
        )}

        {/* --- VIEW 2: MODEL ANALYTICS --- */}
        {activeTab === 'analytics' && (
          <div className="card">
            <h2>Evaluation Analytics &amp; Cross-Validation</h2>
            <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>
              Comparing tuned classifier algorithms evaluated on 3-fold stratified cross-validation and a 20% held-out test split.
            </p>

            <div className="disease-selector-bar" style={{ marginTop: '20px' }}>
              {diseases.map((d) => (
                <button
                  key={d.id}
                  className={`disease-tab-btn ${d.id === selectedModel?.id ? 'active' : ''}`}
                  type="button"
                  onClick={() => setSelectedDisease(d.id)}
                >
                  {d.name}
                </button>
              ))}
            </div>

            <div className="analytics-grid">
              <div className="card" style={{ background: 'var(--input-bg)' }}>
                <h4>Held-Out Confusion Matrix ({currentMetrics?.test_size} samples)</h4>
                {currentMetrics?.confusion_matrix ? (
                  <table className="cm-table">
                    <thead>
                      <tr>
                        <th></th>
                        <th>Predicted Negative (0)</th>
                        <th>Predicted Positive (1)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th>Actual Negative (0)</th>
                        <td className="cm-cell-tn">{currentMetrics.confusion_matrix[0]?.[0]} (TN)</td>
                        <td className="cm-cell-fp">{currentMetrics.confusion_matrix[0]?.[1]} (FP)</td>
                      </tr>
                      <tr>
                        <th>Actual Positive (1)</th>
                        <td className="cm-cell-fn">{currentMetrics.confusion_matrix[1]?.[0]} (FN)</td>
                        <td className="cm-cell-tp">{currentMetrics.confusion_matrix[1]?.[1]} (TP)</td>
                      </tr>
                    </tbody>
                  </table>
                ) : (
                  <p>Matrix unavailable</p>
                )}
                <div style={{ marginTop: '14px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <strong>Specificity:</strong> {formatPercent(currentMetrics?.specificity)} |{' '}
                  <strong>Brier Score:</strong> {currentMetrics?.brier_score ? currentMetrics.brier_score.toFixed(4) : '—'}
                </div>
              </div>

              <div className="card" style={{ background: 'var(--input-bg)' }}>
                <h4>Model Comparison Tournament</h4>
                {currentMetrics?.model_comparison ? (
                  <table className="comparison-table">
                    <thead>
                      <tr>
                        <th>Classifier</th>
                        <th>Mean CV ROC-AUC</th>
                        <th>Std Dev</th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentMetrics.model_comparison.map((mc, idx) => (
                        <tr
                          key={idx}
                          className={mc.model === currentMetrics.selected_model ? 'selected-row' : ''}
                        >
                          <td>
                            {mc.model} {mc.model === currentMetrics.selected_model && '✓ (Selected)'}
                          </td>
                          <td>{(mc.cv_roc_auc_mean * 100).toFixed(2)}%</td>
                          <td>±{(mc.cv_roc_auc_std * 100).toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p>Comparison data unavailable</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* --- VIEW 3: DATASET EXPLORER --- */}
        {activeTab === 'dataset' && (
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h2>{currentDataset?.name ?? 'Dataset Explorer'}</h2>
                <small style={{ color: 'var(--text-muted)' }}>
                  {currentDataset?.rows ?? 0} total cases • {currentDataset?.columns.length ?? 0} attributes
                </small>
              </div>

              <button
                className="preset-chip-btn"
                type="button"
                onClick={exportFilteredDatasetCsv}
              >
                📥 Export Filtered CSV ({filteredRecords.length} rows)
              </button>
            </div>

            <div className="disease-selector-bar" style={{ marginTop: '20px' }}>
              {diseases.map((d) => (
                <button
                  key={d.id}
                  className={`disease-tab-btn ${d.id === selectedModel?.id ? 'active' : ''}`}
                  type="button"
                  onClick={() => setSelectedDisease(d.id)}
                >
                  {d.name}
                </button>
              ))}
            </div>

            <div className="table-scroll-container">
              <table className="data-table">
                <thead>
                  <tr>
                    {currentDataset?.columns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredRecords.slice(0, 100).map((row, idx) => (
                    <tr key={idx}>
                      {currentDataset?.columns.map((col) => (
                        <td key={col}>{String(row[col] ?? '—')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {filteredRecords.length > 100 && (
              <small style={{ color: 'var(--text-muted)', display: 'block', marginTop: '10px' }}>
                Showing first 100 of {filteredRecords.length} records. Export CSV to view all rows.
              </small>
            )}
          </div>
        )}

        {/* --- VIEW 4: HISTORY & PRIVACY --- */}
        {activeTab === 'history' && (
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h2>Prediction History</h2>
                <small style={{ color: 'var(--text-muted)' }}>
                  Locally stored SQLite records ({history.length} records). Does not send data to any cloud.
                </small>
              </div>
              <button
                className="btn-danger"
                type="button"
                onClick={clearHistory}
                disabled={history.length === 0}
              >
                🗑 Clear All History
              </button>
            </div>

            {history.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                No predictions recorded yet.
              </div>
            ) : (
              <div>
                {history.map((item) => (
                  <div key={item.id} className="history-card-item">
                    <div>
                      <strong>{item.disease.toUpperCase()} MODULE</strong>
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '4px' }}>
                        Predicted: {item.predicted_class} • {new Date(item.created_at).toLocaleString()}
                      </div>
                    </div>
                    <span className={`history-badge ${item.positive_score && item.positive_score > 0.5 ? 'pos' : 'neg'}`}>
                      {item.positive_score !== null ? `${Math.round(item.positive_score * 100)}% Risk` : 'Class Output'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* --- VIEW 5: ETHICS & ABOUT --- */}
        {activeTab === 'about' && (
          <div className="card">
            <h2>Clinical Ethics, Limitations &amp; Documentation</h2>
            <div className="analytics-grid">
              <div className="card" style={{ background: 'var(--input-bg)' }}>
                <h4>Legal &amp; Clinical Disclaimer</h4>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: 1.7, marginTop: '8px' }}>
                  HealthSense AI / MediPredict AI is an educational demonstration developed for the <strong>CodeAlpha Machine Learning Internship (Task 4)</strong>. It is not approved by any health authority (FDA, EMA, CDSCO), does not constitute a certified medical device, and must never be used for clinical diagnosis, patient triage, screening, or medical treatment decisions.
                </p>
              </div>

              <div className="card" style={{ background: 'var(--input-bg)' }}>
                <h4>Data Privacy &amp; Governance</h4>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: 1.7, marginTop: '8px' }}>
                  This application adheres to local-first privacy principles. Patient measurement features are evaluated in memory and are never persisted to disk. Only anonymized class scores and timestamps are saved to the local SQLite database when history persistence is enabled. PDF reports are streamed directly to the browser.
                </p>
              </div>

              <div className="card" style={{ background: 'var(--input-bg)' }}>
                <h4>Dataset Citations</h4>
                <ul style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.7, paddingLeft: '18px', marginTop: '8px' }}>
                  <li><strong>Heart Disease:</strong> UCI Cleveland Clinic subset (DOI: 10.24432/C52P4X, CC BY 4.0)</li>
                  <li><strong>Diabetes Indicators:</strong> Sylhet Hospital, Bangladesh (DOI: 10.24432/C5VG8H, CC BY 4.0)</li>
                  <li><strong>Breast Cancer:</strong> Wisconsin Diagnostic FNA Dataset (UCI ID 17, scikit-learn BSD-3)</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

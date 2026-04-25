import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Code2, 
  Play, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  Loader2, 
  Sparkles,
  FileCode,
  TestTube2,
  Wrench,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  AlertCircle,
  Zap,
  Settings,
  History,
  Square
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const PIPELINE_STEPS = [
  { key: 'processing_prompt', label: 'Process Prompt' },
  { key: 'generating_code', label: 'Generate Code' },
  { key: 'executing_tests', label: 'Run Tests' },
  { key: 'repairing_code', label: 'Repair (if needed)' },
];

function getStepIndex(step) {
  const map = {
    processing_prompt: 0,
    generating_code: 1,
    generating_tests: 2,
    executing_tests: 2,
    analyzing_errors: 3,
    repairing_code: 3,
    saving_output: 3,
    completed: 4,
    failed: -1,
  };
  return map[step] ?? -1;
}

function parseSSELines(text) {
  const events = [];
  const lines = text.split('\n');
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      try {
        events.push(JSON.parse(line.slice(6)));
      } catch {}
    }
  }
  return events;
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [settings, setSettings] = useState({
    maxIterations: 3,
    model: 'gemma4:e4b',
    outputDir: 'real'
  });
  const [expandedSections, setExpandedSections] = useState({
    spec: true,
    code: true,
    tests: false
  });
  const [streamProgress, setStreamProgress] = useState({ step: '', message: '', progress: 0 });

  const abortRef = useRef(null);

  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  // Fetch job history
  useEffect(() => {
    if (showHistory) {
      fetchJobs();
    }
  }, [showHistory]);

  const fetchJobs = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/jobs`);
      const data = await response.json();
      setJobs(data.jobs || []);
    } catch (err) {
      console.error('Error fetching jobs:', err);
    }
  };

  const handleCancel = useCallback(async () => {
    if (abortRef.current) abortRef.current.abort();

    if (currentJobId) {
      try {
        const res = await fetch(`${API_BASE}/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
        if (res.status === 409) {
          const body = await res.json();
          setError(body.detail || 'Job already finished');
        }
      } catch {}
    }

    setIsGenerating(false);
    setStreamProgress({ step: '', message: '', progress: 0 });
  }, [currentJobId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setIsGenerating(true);
    setError(null);
    setResult(null);
    setCurrentJobId(null);
    setStreamProgress({ step: 'starting', message: 'Connecting...', progress: 0 });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/api/generate/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          max_iterations: settings.maxIterations,
          model: settings.model,
          output_dir: settings.outputDir
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        if (response.status === 429) {
          throw new Error(body?.detail || 'Rate limit exceeded. Please wait a moment and try again.');
        }
        if (response.status === 422) {
          const details = body?.detail;
          if (Array.isArray(details)) {
            const msgs = details.map(d => `${d.loc?.slice(-1)?.[0] || 'field'}: ${d.msg}`).join('; ');
            throw new Error(`Validation error — ${msgs}`);
          }
          throw new Error(body?.detail || 'Validation error');
        }
        throw new Error(body?.detail || `Server error (${response.status})`);
      }

      const jobId = response.headers.get('X-Job-ID');
      if (jobId) setCurrentJobId(jobId);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          const events = parseSSELines(part + '\n');
          for (const evt of events) {
            if (evt.job_id && !currentJobId) {
              setCurrentJobId(evt.job_id);
            }

            setStreamProgress({
              step: evt.step || '',
              message: evt.message || '',
              progress: evt.progress ?? 0
            });

            if (evt.step === 'completed' && evt.result) {
              setResult(evt.result);
              setIsGenerating(false);
              setStreamProgress({ step: 'completed', message: 'Done', progress: 1 });
            } else if (evt.step === 'failed' || evt.status === 'failed') {
              setError(evt.message || 'Pipeline failed');
              setIsGenerating(false);
            }
          }
        }
      }

      if (buffer.trim()) {
        const events = parseSSELines(buffer);
        for (const evt of events) {
          if (evt.step === 'completed' && evt.result) {
            setResult(evt.result);
            setIsGenerating(false);
          } else if (evt.step === 'failed' || evt.status === 'failed') {
            setError(evt.message || 'Pipeline failed');
            setIsGenerating(false);
          }
        }
      }

      setIsGenerating(false);
    } catch (err) {
      if (err.name === 'AbortError') return;
      setError(err.message);
      setIsGenerating(false);
    } finally {
      abortRef.current = null;
    }
  };

  const copyToClipboard = async (text) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const loadHistoryJob = async (job) => {
    if (job.result) {
      setResult(job.result);
    } else if (job.status === 'completed') {
      try {
        const res = await fetch(`${API_BASE}/api/jobs/${job.job_id}`);
        const data = await res.json();
        if (data.result) setResult(data.result);
      } catch {}
    }
    setPrompt(job.prompt);
    setCurrentJobId(job.job_id);
    setShowHistory(false);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="w-5 h-5 text-emerald-500" />;
      case 'failed': return <XCircle className="w-5 h-5 text-red-500" />;
      case 'running': return <Loader2 className="w-5 h-5 text-sky-500 animate-spin" />;
      case 'cancelled': return <Square className="w-5 h-5 text-slate-400" />;
      default: return <Clock className="w-5 h-5 text-amber-500" />;
    }
  };

  const activeStepIndex = getStepIndex(streamProgress.step);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-sky-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/25">
                <Code2 className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-slate-800">Code Synthesis</h1>
                <p className="text-xs text-slate-500">AI-Powered Code Generation</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className={`p-2 rounded-lg transition-colors ${showHistory ? 'bg-sky-100 text-sky-600' : 'hover:bg-slate-100 text-slate-600'}`}
              >
                <History className="w-5 h-5" />
              </button>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`p-2 rounded-lg transition-colors ${showSettings ? 'bg-sky-100 text-sky-600' : 'hover:bg-slate-100 text-slate-600'}`}
              >
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Settings Panel */}
        {showSettings && (
          <div className="mb-6 bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">Settings</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Model
                </label>
                <select
                  value={settings.model}
                  onChange={(e) => setSettings({ ...settings, model: e.target.value })}
                  className="w-full px-4 py-2 rounded-lg border border-slate-300 bg-white focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
                >
                  <option value="gemma4:e4b">Gemma 4B</option>
                  <option value="llama3.1:8b">Llama 3.1 8B</option>
                  <option value="qwen3:14b">Qwen3 14B</option>
                  <option value="codellama:13b">CodeLlama 13B</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Max Repair Iterations
                </label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={settings.maxIterations}
                  onChange={(e) => setSettings({ ...settings, maxIterations: parseInt(e.target.value) })}
                  className="w-full px-4 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
                />
              </div>
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Output Directory
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={settings.outputDir}
                    onChange={(e) => setSettings({ ...settings, outputDir: e.target.value })}
                    placeholder="e.g., real, output, src/generated"
                    className="flex-1 px-4 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 font-mono text-sm"
                  />
                  <button
                    type="button"
                    onClick={() => setSettings({ ...settings, outputDir: 'real' })}
                    className="px-3 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-600 transition-colors"
                  >
                    Reset
                  </button>
                </div>
                <p className="mt-1 text-xs text-slate-500">Relative to project root. The directory will be created if it doesn't exist.</p>
              </div>
            </div>
          </div>
        )}

        {/* History Panel */}
        {showHistory && (
          <div className="mb-6 bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">History</h3>
            {jobs.length === 0 ? (
              <p className="text-slate-500 text-center py-4">No previous jobs</p>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {jobs.map((job) => (
                  <button
                    key={job.job_id}
                    onClick={() => loadHistoryJob(job)}
                    className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-slate-50 transition-colors text-left"
                  >
                    {getStatusIcon(job.status)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-slate-800 truncate">{job.prompt}</p>
                      <p className="text-xs text-slate-500">{new Date(job.created_at).toLocaleString()}</p>
                    </div>
                    {job.duration_seconds != null && (
                      <span className="text-xs text-slate-400 flex-shrink-0">{job.duration_seconds}s</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Input Section */}
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm mb-8">
          <form onSubmit={handleSubmit}>
            <label className="block text-sm font-medium text-slate-700 mb-3">
              Describe the program you want to create
            </label>
            <div className="relative">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="e.g., Create an LRU cache implementation with get and put methods..."
                className="w-full h-32 px-4 py-3 rounded-xl border border-slate-300 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 resize-none text-slate-800 placeholder:text-slate-400"
                disabled={isGenerating}
              />
              <div className="absolute bottom-3 right-3">
                <Sparkles className="w-5 h-5 text-slate-300" />
              </div>
            </div>
            <div className="mt-4 flex items-center justify-between">
              <p className="text-sm text-slate-500">
                Press <kbd className="px-2 py-0.5 bg-slate-100 rounded text-xs font-medium">Enter</kbd> or click Generate
              </p>
              <div className="flex items-center gap-2">
                {isGenerating && (
                  <button
                    type="button"
                    onClick={handleCancel}
                    className="flex items-center gap-2 px-4 py-2.5 bg-red-50 hover:bg-red-100 text-red-600 rounded-xl font-medium border border-red-200 transition-colors"
                  >
                    <XCircle className="w-5 h-5" />
                    Cancel
                  </button>
                )}
                <button
                  type="submit"
                  disabled={isGenerating || !prompt.trim()}
                  className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-sky-500 to-indigo-600 text-white rounded-xl font-medium shadow-lg shadow-sky-500/25 hover:shadow-xl hover:shadow-sky-500/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      Generate
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* Progress Indicator */}
        {isGenerating && (
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm mb-8">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-sky-100 flex items-center justify-center">
                <Loader2 className="w-6 h-6 text-sky-600 animate-spin" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-slate-800">Processing your request</h3>
                <p className="text-sm text-slate-500">
                  {streamProgress.message || 'Initializing pipeline...'}
                </p>
              </div>
            </div>
            <div className="mt-4 w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-sky-500 to-indigo-500 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${Math.max(streamProgress.progress * 100, 2)}%` }}
              />
            </div>
            <div className="mt-4 grid grid-cols-4 gap-2">
              {PIPELINE_STEPS.map((step, i) => {
                const isActive = i === activeStepIndex;
                const isDone = i < activeStepIndex;
                const stepClass = isDone
                  ? 'bg-emerald-50 text-emerald-700'
                  : isActive
                    ? 'bg-sky-50 text-sky-700'
                    : 'bg-slate-50 text-slate-500';
                const dotClass = isDone
                  ? 'bg-emerald-500 text-white'
                  : isActive
                    ? 'bg-sky-500 text-white'
                    : 'bg-slate-200 text-slate-600';

                return (
                  <div key={step.key} className={`flex items-center gap-2 p-2 rounded-lg ${stepClass}`}>
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${dotClass}`}>
                      {isDone ? <Check className="w-3.5 h-3.5" /> : i + 1}
                    </div>
                    <span className="text-xs font-medium">{step.label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-2xl p-6 mb-8">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-lg font-semibold text-red-800">
                  {error.includes('Rate limit') ? 'Rate Limit Exceeded' :
                   error.includes('Validation') ? 'Invalid Input' :
                   'Generation Failed'}
                </h3>
                <p className="text-sm text-red-600 mt-1">{error}</p>
                {error.includes('Rate limit') && (
                  <p className="text-xs text-red-500 mt-2">Please wait a minute before trying again.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Status Card */}
            <div className={`rounded-2xl border p-6 shadow-sm ${result.success ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
              <div className="flex items-center gap-4">
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${result.success ? 'bg-emerald-500' : 'bg-amber-500'}`}>
                  {result.success ? (
                    <CheckCircle2 className="w-8 h-8 text-white" />
                  ) : (
                    <AlertCircle className="w-8 h-8 text-white" />
                  )}
                </div>
                <div className="flex-1">
                  <h3 className={`text-xl font-semibold ${result.success ? 'text-emerald-800' : 'text-amber-800'}`}>
                    {result.success ? 'Code Generated Successfully!' : 'Generation Completed with Issues'}
                  </h3>
                  <div className="flex items-center gap-4 mt-1">
                    <span className="text-sm text-slate-600">
                      {result.test_results?.filter(t => t.passed).length || 0}/{result.test_results?.length || 0} tests passed
                    </span>
                    <span className="text-sm text-slate-600">
                      {result.repair_iterations} repair iterations
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Problem Specification */}
            {result.problem_spec && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <button
                  onClick={() => toggleSection('spec')}
                  className="w-full flex items-center justify-between p-4 hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center">
                      <FileCode className="w-5 h-5 text-purple-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-800">Problem Specification</h3>
                  </div>
                  {expandedSections.spec ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                </button>
                {expandedSections.spec && (
                  <div className="px-4 pb-4 border-t border-slate-100">
                    <div className="mt-4 space-y-3">
                      <div>
                        <span className="text-sm font-medium text-slate-500">Summary</span>
                        <p className="text-slate-800">{result.problem_spec.problem_summary}</p>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <span className="text-sm font-medium text-slate-500">Inputs</span>
                          <ul className="list-disc list-inside text-slate-700 text-sm">
                            {result.problem_spec.inputs?.map((inp, i) => <li key={i}>{inp}</li>)}
                          </ul>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-slate-500">Outputs</span>
                          <ul className="list-disc list-inside text-slate-700 text-sm">
                            {result.problem_spec.outputs?.map((out, i) => <li key={i}>{out}</li>)}
                          </ul>
                        </div>
                      </div>
                      {result.problem_spec.constraints?.length > 0 && (
                        <div>
                          <span className="text-sm font-medium text-slate-500">Constraints</span>
                          <ul className="list-disc list-inside text-slate-700 text-sm">
                            {result.problem_spec.constraints.map((c, i) => <li key={i}>{c}</li>)}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Generated Code */}
            {result.final_code && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <button
                  onClick={() => toggleSection('code')}
                  className="w-full flex items-center justify-between p-4 hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-sky-100 flex items-center justify-center">
                      <Code2 className="w-5 h-5 text-sky-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-800">Generated Code</h3>
                  </div>
                  {expandedSections.code ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                </button>
                {expandedSections.code && (
                  <div className="px-4 pb-4 border-t border-slate-100">
                    <div className="relative mt-4">
                      <button
                        onClick={() => copyToClipboard(result.final_code)}
                        className="absolute top-3 right-3 p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors z-10"
                      >
                        {copied ? (
                          <Check className="w-4 h-4 text-emerald-400" />
                        ) : (
                          <Copy className="w-4 h-4 text-slate-300" />
                        )}
                      </button>
                      <pre className="code-block max-h-96 overflow-auto">
                        <code>{result.final_code}</code>
                      </pre>
                    </div>
                    {result.output_file && (
                      <p className="mt-3 text-sm text-slate-500">
                        Saved to: <code className="px-2 py-0.5 bg-slate-100 rounded text-slate-700">{result.output_file}</code>
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Test Results */}
            {result.test_results && result.test_results.length > 0 && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <button
                  onClick={() => toggleSection('tests')}
                  className="w-full flex items-center justify-between p-4 hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-emerald-100 flex items-center justify-center">
                      <TestTube2 className="w-5 h-5 text-emerald-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-800">Test Results</h3>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      result.test_results.every(t => t.passed) 
                        ? 'bg-emerald-100 text-emerald-700' 
                        : 'bg-amber-100 text-amber-700'
                    }`}>
                      {result.test_results.filter(t => t.passed).length}/{result.test_results.length} passed
                    </span>
                  </div>
                  {expandedSections.tests ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                </button>
                {expandedSections.tests && (
                  <div className="px-4 pb-4 border-t border-slate-100">
                    <div className="mt-4 space-y-2">
                      {result.test_results.map((test, i) => (
                        <div
                          key={i}
                          className={`p-3 rounded-lg border ${test.passed ? 'bg-emerald-50 border-emerald-200' : 'bg-red-50 border-red-200'}`}
                        >
                          <div className="flex items-center gap-2">
                            {test.passed ? (
                              <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                            ) : (
                              <XCircle className="w-5 h-5 text-red-500" />
                            )}
                            <span className={`font-medium ${test.passed ? 'text-emerald-800' : 'text-red-800'}`}>
                              {test.test_id}
                            </span>
                          </div>
                          {!test.passed && (
                            <div className="mt-2 text-sm">
                              <p className="text-slate-600">Expected: <code className="bg-white px-1 rounded">{test.expected}</code></p>
                              <p className="text-slate-600">Got: <code className="bg-white px-1 rounded">{test.actual}</code></p>
                              {test.error && <p className="text-red-600 mt-1">{test.error}</p>}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error Analysis */}
            {result.error_analysis && !result.success && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-amber-100 flex items-center justify-center">
                      <Wrench className="w-5 h-5 text-amber-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-800">Error Analysis</h3>
                  </div>
                  <div className="mt-4 space-y-3">
                    {result.error_analysis.error_summary && (
                      <div>
                        <span className="text-sm font-medium text-slate-500">Summary</span>
                        <p className="text-slate-800">{result.error_analysis.error_summary}</p>
                      </div>
                    )}
                    {result.error_analysis.root_causes?.length > 0 && (
                      <div>
                        <span className="text-sm font-medium text-slate-500">Root Causes</span>
                        <ul className="list-disc list-inside text-slate-700">
                          {result.error_analysis.root_causes.map((cause, i) => (
                            <li key={i}>{cause}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {!isGenerating && !result && !error && (
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br from-sky-100 to-indigo-100 flex items-center justify-center mb-6">
              <Sparkles className="w-10 h-10 text-sky-500" />
            </div>
            <h3 className="text-xl font-semibold text-slate-800 mb-2">Ready to Generate Code</h3>
            <p className="text-slate-500 max-w-md mx-auto">
              Describe your program in natural language and our AI will generate, test, and refine the code for you.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white/50 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-slate-500">
            Code Synthesis Pipeline • AI-Powered Code Generation
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

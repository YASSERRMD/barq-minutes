import { AlertTriangle, Brain, Mic, RefreshCw, Sparkles } from 'lucide-react';
import { useModelBoot } from '../context/ModelBootContext';
import type { ModelKey } from '../context/ModelBootContext';

const MODEL_META: Record<ModelKey, { label: string; sub: string; Icon: React.ElementType }> = {
  llm: {
    label: 'glm5.1-distill (LLM)',
    sub: 'Meeting summaries, decisions, action items · Q4 quantised',
    Icon: Brain,
  },
  asr: {
    label: 'LFM2.5-Audio 1.5B (ASR)',
    sub: 'Local speech-to-text · runs entirely in your browser',
    Icon: Mic,
  },
  embeddings: {
    label: 'MiniLM-L6 (Embeddings)',
    sub: 'Semantic search over your transcripts · Q8',
    Icon: Sparkles,
  },
};

function ProgressBar({ value }: { value: number | null }) {
  const pct = value == null ? null : Math.max(0, Math.min(100, value));
  return (
    <div className="boot-bar-track" role="progressbar" aria-valuenow={pct ?? undefined} aria-valuemin={0} aria-valuemax={100}>
      <div
        className={`boot-bar-fill ${pct == null ? 'boot-bar-indeterminate' : ''}`}
        style={pct != null ? { width: `${pct}%` } : undefined}
      />
    </div>
  );
}

export default function ModelBootSplash() {
  const { state, anyError, retry } = useModelBoot();

  const models: ModelKey[] = ['llm', 'asr', 'embeddings'];

  return (
    <div className="boot-splash" role="status" aria-live="polite">
      <div className="boot-card">
        {/* Branding */}
        <header className="boot-header">
          <span className="boot-brand-mark" />
          <span className="boot-brand-name">barq-minutes</span>
        </header>

        <p className="boot-headline">Loading AI models locally</p>
        <p className="boot-sub">
          All processing happens in your browser. No data leaves your device.
          Large models are cached after first download.
        </p>

        {/* Per-model rows */}
        <div className="boot-model-list">
          {models.map((key) => {
            const m = state[key];
            const { label, sub, Icon } = MODEL_META[key];
            const isError = m.status === 'error';
            const isReady = m.status === 'ready';

            return (
              <div key={key} className={`boot-model-row ${m.status}`}>
                <div className="boot-model-icon">
                  <Icon size={18} />
                </div>
                <div className="boot-model-info">
                  <div className="boot-model-header">
                    <span className="boot-model-label">{label}</span>
                    <span className={`boot-model-badge ${m.status}`}>
                      {isError ? 'Error' : isReady ? 'Ready' : m.progress != null ? `${Math.round(m.progress)}%` : '…'}
                    </span>
                  </div>
                  <span className="boot-model-sub">{sub}</span>
                  <ProgressBar value={m.progress} />
                  {m.message && (
                    <p className="boot-model-message">
                      {isError && <AlertTriangle size={13} />}
                      {m.message}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Error retry */}
        {anyError && (
          <div className="boot-error-actions">
            <p className="boot-error-text">
              One or more models failed to load. Check your connection and try again.
            </p>
            <button className="button primary" type="button" onClick={retry}>
              <RefreshCw size={16} />
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

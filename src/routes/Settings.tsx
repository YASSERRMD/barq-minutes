import { useEffect, useState } from 'react';
import { Database, Trash2 } from 'lucide-react';
import { clearAudioBlobs } from '../storage/audio';
import { clearMeetings } from '../storage/meetings';
import { MODEL_IDS } from '../models/modelConfig';
import { formatStorageBytes } from '../utils/time';

type Usage = {
  usage: number;
  quota: number;
};

export default function Settings() {
  const [usage, setUsage] = useState<Usage | null>(null);
  const [status, setStatus] = useState('');

  async function refreshUsage() {
    if (!navigator.storage?.estimate) return;
    const estimate = await navigator.storage.estimate();
    setUsage({
      usage: estimate.usage ?? 0,
      quota: estimate.quota ?? 0,
    });
  }

  useEffect(() => {
    refreshUsage();
  }, []);

  async function clearAllData() {
    setStatus('Clearing IndexedDB records');
    await Promise.all([clearMeetings(), clearAudioBlobs()]);
    await refreshUsage();
    setStatus('Local meeting and audio records cleared');
  }

  return (
    <section>
      <header className="page-header">
        <div>
          <p className="page-kicker">Settings</p>
          <h1 className="page-title">Local runtime</h1>
          <p className="page-subtitle">
            Inspect model configuration, browser isolation, and local storage usage.
          </p>
        </div>
      </header>

      <div className="settings-grid">
        <section className="panel">
          <h2 className="section-title">Models</h2>
          <dl className="settings-list">
            <div>
              <dt>LLM</dt>
              <dd>{MODEL_IDS.llm}</dd>
            </div>
            <div>
              <dt>ASR</dt>
              <dd>{MODEL_IDS.asr}</dd>
            </div>
            <div>
              <dt>Embeddings</dt>
              <dd>{MODEL_IDS.embeddings}</dd>
            </div>
            <div>
              <dt>SharedArrayBuffer</dt>
              <dd>{typeof SharedArrayBuffer === 'undefined' ? 'Unavailable' : 'Available'}</dd>
            </div>
          </dl>
        </section>

        <section className="panel">
          <h2 className="section-title">IndexedDB usage</h2>
          <div className="usage-card">
            <Database size={20} />
            <span>{usage ? `${formatStorageBytes(usage.usage)} of ${formatStorageBytes(usage.quota)}` : 'Unavailable'}</span>
          </div>
          <button className="button danger" type="button" onClick={clearAllData}>
            <Trash2 size={17} />
            Clear all data
          </button>
          {status ? <p className="status-line">{status}</p> : null}
        </section>
      </div>
    </section>
  );
}

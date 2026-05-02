import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import { loadAsrSession, type AsrSession } from '../models/asrSession';
import { loadEmbeddingSession, type EmbeddingSession } from '../models/embedSession';
import { loadLlmSession, type LlmSession } from '../models/llmSession';
import type { ModelLoadProgress } from '../models/modelConfig';

export type ModelKey = 'llm' | 'asr' | 'embeddings';

export type ModelState = ModelLoadProgress & {
  /** Resolved session handle. Null until the model is ready. */
  session: LlmSession | AsrSession | EmbeddingSession | null;
};

type ModelBootState = {
  llm: ModelState;
  asr: ModelState;
  embeddings: ModelState;
};

type ModelBootContextValue = {
  state: ModelBootState;
  allReady: boolean;
  anyError: boolean;
  retry: () => void;
  llmSession: LlmSession | null;
  asrSession: AsrSession | null;
  embeddingSession: EmbeddingSession | null;
};

const idle = (model: ModelKey): ModelState => ({
  status: 'idle',
  model,
  message: 'Waiting...',
  progress: 0,
  session: null,
});

const ModelBootContext = createContext<ModelBootContextValue | null>(null);

export function ModelBootProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ModelBootState>({
    llm: idle('llm'),
    asr: idle('asr'),
    embeddings: idle('embeddings'),
  });

  // Track sessions separately so TypeScript can type them cleanly.
  const [llmSession, setLlmSession] = useState<LlmSession | null>(null);
  const [asrSession, setAsrSession] = useState<AsrSession | null>(null);
  const [embeddingSession, setEmbeddingSession] = useState<EmbeddingSession | null>(null);

  // Prevent double-loading in StrictMode.
  const loadingRef = useRef(false);

  const setModelProgress = useCallback(
    (key: ModelKey) =>
      (progress: ModelLoadProgress) => {
        setState((prev) => ({
          ...prev,
          [key]: { ...prev[key], ...progress },
        }));
      },
    [],
  );

  const load = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;

    setState({
      llm: { ...idle('llm'), status: 'loading', message: 'Waiting for ASR WebGPU setup' },
      asr: { ...idle('asr'), status: 'loading', message: 'Initializing...' },
      embeddings: { ...idle('embeddings'), status: 'loading', message: 'Initializing...' },
    });

    const embeddingTask = loadEmbeddingSession(setModelProgress('embeddings'))
      .then((session) => {
        setEmbeddingSession(session);
        setState((prev) => ({
          ...prev,
          embeddings: { ...prev.embeddings, session, status: 'ready', progress: 100 },
        }));
      })
      .catch((err) => {
        setState((prev) => ({
          ...prev,
          embeddings: {
            ...prev.embeddings,
            status: 'error',
            message: err instanceof Error ? err.message : String(err),
            progress: null,
          },
        }));
      });

    await loadAsrSession(setModelProgress('asr'))
      .then((session) => {
        setAsrSession(session);
        setState((prev) => ({
          ...prev,
          asr: { ...prev.asr, session, status: 'ready', progress: 100 },
          llm: prev.llm.status === 'loading'
            ? { ...prev.llm, message: 'Starting LLM WebGPU setup' }
            : prev.llm,
        }));
      })
      .catch((err) => {
        setState((prev) => ({
          ...prev,
          asr: {
            ...prev.asr,
            status: 'error',
            message: err instanceof Error ? err.message : String(err),
            progress: null,
          },
          llm: prev.llm.status === 'loading'
            ? { ...prev.llm, message: 'Starting LLM WebGPU setup' }
            : prev.llm,
        }));
      });

    await loadLlmSession(setModelProgress('llm'))
      .then((session) => {
        setLlmSession(session);
        setState((prev) => ({
          ...prev,
          llm: { ...prev.llm, session, status: 'ready', progress: 100 },
        }));
      })
      .catch((err) => {
        setState((prev) => ({
          ...prev,
          llm: {
            ...prev.llm,
            status: 'error',
            message: err instanceof Error ? err.message : String(err),
            progress: null,
          },
        }));
      });

    await embeddingTask;
  }, [setModelProgress]);

  const retry = useCallback(() => {
    loadingRef.current = false;
    void load();
  }, [load]);

  useEffect(() => {
    void load();
  }, [load]);

  const allReady =
    state.llm.status === 'ready' &&
    state.asr.status === 'ready' &&
    state.embeddings.status === 'ready';

  const anyError =
    state.llm.status === 'error' ||
    state.asr.status === 'error' ||
    state.embeddings.status === 'error';

  return (
    <ModelBootContext.Provider
      value={{ state, allReady, anyError, retry, llmSession, asrSession, embeddingSession }}
    >
      {children}
    </ModelBootContext.Provider>
  );
}

export function useModelBoot() {
  const ctx = useContext(ModelBootContext);
  if (!ctx) throw new Error('useModelBoot must be used inside <ModelBootProvider>');
  return ctx;
}

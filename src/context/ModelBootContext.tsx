import {
  createContext,
  useCallback,
  useContext,
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
  loadAll: () => Promise<void>;
  ensureAsrSession: () => Promise<AsrSession>;
  ensureLlmSession: () => Promise<LlmSession>;
  ensureEmbeddingSession: () => Promise<EmbeddingSession>;
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

  const [llmSession, setLlmSession] = useState<LlmSession | null>(null);
  const [asrSession, setAsrSession] = useState<AsrSession | null>(null);
  const [embeddingSession, setEmbeddingSession] = useState<EmbeddingSession | null>(null);

  const llmRef = useRef<LlmSession | null>(null);
  const asrRef = useRef<AsrSession | null>(null);
  const embeddingRef = useRef<EmbeddingSession | null>(null);
  const llmLoadingRef = useRef<Promise<LlmSession> | null>(null);
  const asrLoadingRef = useRef<Promise<AsrSession> | null>(null);
  const embeddingLoadingRef = useRef<Promise<EmbeddingSession> | null>(null);

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

  const ensureAsrSession = useCallback(async () => {
    if (asrRef.current) return asrRef.current;
    if (asrLoadingRef.current) return asrLoadingRef.current;

    setState((prev) => ({
      ...prev,
      asr: { ...prev.asr, status: 'loading', message: 'Preparing ASR model', progress: prev.asr.progress ?? 0 },
    }));

    const promise = loadAsrSession(setModelProgress('asr'))
      .then((session) => {
        asrRef.current = session;
        setAsrSession(session);
        setState((prev) => ({
          ...prev,
          asr: { ...prev.asr, session, status: 'ready', message: `ASR ready on ${session.backend}`, progress: 100 },
        }));
        return session;
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
        }));
        throw err;
      })
      .finally(() => {
        asrLoadingRef.current = null;
      });

    asrLoadingRef.current = promise;
    return promise;
  }, [setModelProgress]);

  const ensureLlmSession = useCallback(async () => {
    if (llmRef.current) return llmRef.current;
    if (llmLoadingRef.current) return llmLoadingRef.current;

    setState((prev) => ({
      ...prev,
      llm: { ...prev.llm, status: 'loading', message: 'Preparing LLM model', progress: prev.llm.progress ?? 0 },
    }));

    const promise = loadLlmSession(setModelProgress('llm'))
      .then((session) => {
        llmRef.current = session;
        setLlmSession(session);
        setState((prev) => ({
          ...prev,
          llm: { ...prev.llm, session, status: 'ready', message: 'LLM ready', progress: 100 },
        }));
        return session;
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
        throw err;
      })
      .finally(() => {
        llmLoadingRef.current = null;
      });

    llmLoadingRef.current = promise;
    return promise;
  }, [setModelProgress]);

  const ensureEmbeddingSession = useCallback(async () => {
    if (embeddingRef.current) return embeddingRef.current;
    if (embeddingLoadingRef.current) return embeddingLoadingRef.current;

    setState((prev) => ({
      ...prev,
      embeddings: { ...prev.embeddings, status: 'loading', message: 'Preparing embedding model', progress: prev.embeddings.progress ?? 0 },
    }));

    const promise = loadEmbeddingSession(setModelProgress('embeddings'))
      .then((session) => {
        embeddingRef.current = session;
        setEmbeddingSession(session);
        setState((prev) => ({
          ...prev,
          embeddings: { ...prev.embeddings, session, status: 'ready', message: 'Embeddings ready', progress: 100 },
        }));
        return session;
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
        throw err;
      })
      .finally(() => {
        embeddingLoadingRef.current = null;
      });

    embeddingLoadingRef.current = promise;
    return promise;
  }, [setModelProgress]);

  const loadAll = useCallback(async () => {
    await ensureAsrSession();
    await ensureLlmSession();
    await ensureEmbeddingSession();
  }, [ensureAsrSession, ensureEmbeddingSession, ensureLlmSession]);

  const retry = useCallback(() => {
    void loadAll();
  }, [loadAll]);

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
      value={{
        state,
        allReady,
        anyError,
        retry,
        loadAll,
        ensureAsrSession,
        ensureLlmSession,
        ensureEmbeddingSession,
        llmSession,
        asrSession,
        embeddingSession,
      }}
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

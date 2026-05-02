export const MODEL_IDS = {
  llm: 'yasserrmd/glm5.1-distill-onnx',
  asr: 'LiquidAI/LFM2.5-Audio-1.5B-ONNX',
  embeddings: 'Xenova/all-MiniLM-L6-v2',
} as const;

export const MODEL_DTYPES = {
  llm: 'q4',
  asr: 'q4',
  embeddings: 'q8',
} as const;

export const MODEL_CACHE_KEYS = {
  llm: 'barq-minutes:glm5.1-distill:q4',
  asr: 'barq-minutes:lfm2.5-audio:q4',
  embeddings: 'barq-minutes:minilm:q8',
} as const;

export const LLM_CONTEXT_TOKENS = 4096;
export const LLM_INPUT_BUDGET_TOKENS = 3400;
export const STRUCTURED_MAX_NEW_TOKENS = 256;
export const SUMMARY_MAX_NEW_TOKENS = 384;

export type ModelLoadProgress = {
  status: 'idle' | 'loading' | 'ready' | 'error';
  model: keyof typeof MODEL_IDS;
  message: string;
  progress: number | null;
};

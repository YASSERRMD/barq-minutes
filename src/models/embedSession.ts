import { pipeline, env } from '@huggingface/transformers';
import { MODEL_DTYPES, MODEL_IDS, type ModelLoadProgress } from './modelConfig';

type EmbeddingProgress = (progress: ModelLoadProgress) => void;

export type EmbeddingSession = {
  embed(texts: string[]): Promise<Float32Array[]>;
};

let embedPromise: Promise<EmbeddingSession> | null = null;

env.allowLocalModels = false;
env.useBrowserCache = true;
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
}

function report(
  onProgress: EmbeddingProgress | undefined,
  message: string,
  progress: number | null,
  status: ModelLoadProgress['status'] = 'loading',
) {
  onProgress?.({
    status,
    model: 'embeddings',
    message,
    progress,
  });
}

function toFloatRows(output: any): Float32Array[] {
  const rows = output.tolist?.() ?? [];
  return rows.map((row: number[]) => Float32Array.from(row));
}

export async function loadEmbeddingSession(
  onProgress?: EmbeddingProgress,
): Promise<EmbeddingSession> {
  if (embedPromise) return embedPromise;

  embedPromise = (async () => {
    try {
      report(onProgress, 'Loading MiniLM embedding model', 0);
      const extractor = await pipeline('feature-extraction', MODEL_IDS.embeddings, {
        dtype: MODEL_DTYPES.embeddings,
        progress_callback: (event: any) => {
          if (event.status === 'progress') {
            report(onProgress, `Loading ${event.file ?? 'embedding model'} from model cache`, event.progress ?? null);
          }
        },
      } as any);

      report(onProgress, 'MiniLM embeddings ready', 100, 'ready');

      return {
        async embed(texts: string[]) {
          if (texts.length === 0) return [];
          const output = await extractor(texts, {
            pooling: 'mean',
            normalize: true,
          });
          return toFloatRows(output);
        },
      };
    } catch (error) {
      embedPromise = null;
      report(
        onProgress,
        error instanceof Error ? error.message : String(error),
        null,
        'error',
      );
      throw error;
    }
  })();

  return embedPromise;
}

export function resetEmbeddingSessionForTests() {
  embedPromise = null;
}

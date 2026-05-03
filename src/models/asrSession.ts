import * as ort from 'onnxruntime-web/webgpu';
import { AutoTokenizer } from '@huggingface/transformers';
import { MODEL_DTYPES, MODEL_IDS, type ModelLoadProgress } from './modelConfig';

type AsrProgress = (progress: ModelLoadProgress) => void;

export type AsrSession = {
  tokenizer: any;
  decoder: ort.InferenceSession;
  audioEncoder: ort.InferenceSession;
  embedTokens: Float32Array;
  embedShape: { vocabSize: number; hiddenSize: number };
  modelBaseUrl: string;
  backend: 'webgpu' | 'wasm';
  layerTypes: string[];
  hiddenSize: number;
  numKVHeads: number;
  headDim: number;
  convL: number;
  vocabSize: number;
};

let asrPromise: Promise<AsrSession> | null = null;
const ASR_CACHE_NAME = `barq-minutes:asr:${MODEL_IDS.asr}:${MODEL_DTYPES.asr}`;

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isOrtSessionCreationRace(error: unknown) {
  const message = String(error);
  return message.includes('another WebGPU EP inference session is being created')
    || message.includes("multiple calls to 'initWasm()' detected")
    || message.includes('no available backend found');
}

function modelBaseUrl() {
  return `https://huggingface.co/${MODEL_IDS.asr}/resolve/main`;
}

function report(onProgress: AsrProgress | undefined, message: string, progress: number | null) {
  onProgress?.({
    status: 'loading',
    model: 'asr',
    message,
    progress,
  });
}

async function openModelCache(): Promise<Cache | null> {
  if (!('caches' in window)) return null;
  try {
    return await caches.open(ASR_CACHE_NAME);
  } catch (error) {
    console.warn('[ASR] Browser cache unavailable', error);
    return null;
  }
}

async function fetchBinaryWithProgress(
  url: string,
  label: string,
  progressStart: number,
  progressEnd: number,
  onProgress?: AsrProgress,
): Promise<Uint8Array> {
  const cache = await openModelCache();
  const cachedResponse = await cache?.match(url);
  if (cachedResponse) {
    report(onProgress, `Loading ${label} from browser cache`, progressStart);
    const buffer = await cachedResponse.arrayBuffer();
    report(onProgress, `Loaded ${label} from browser cache`, progressEnd);
    return new Uint8Array(buffer);
  }

  report(onProgress, `Downloading ${label}`, progressStart);
  const response = await fetch(url, { credentials: 'omit' });
  if (!response.ok) {
    throw new Error(`Failed to download ${label}: ${response.status}`);
  }

  const total = Number(response.headers.get('content-length') ?? 0);
  const reader = response.body?.getReader();
  if (!reader) {
    const buffer = await response.arrayBuffer();
    await cache?.put(url, new Response(buffer.slice(0), {
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'application/octet-stream',
        'Content-Length': String(buffer.byteLength),
      },
    }));
    report(onProgress, `Downloaded ${label}`, progressEnd);
    return new Uint8Array(buffer);
  }

  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    chunks.push(value);
    loaded += value.byteLength;

    if (total > 0) {
      const fileProgress = loaded / total;
      const progress = progressStart + (progressEnd - progressStart) * fileProgress;
      const loadedMb = (loaded / 1024 / 1024).toFixed(1);
      const totalMb = (total / 1024 / 1024).toFixed(1);
      report(onProgress, `Downloading ${label} (${loadedMb}/${totalMb} MB)`, progress);
    } else {
      const loadedMb = (loaded / 1024 / 1024).toFixed(1);
      report(onProgress, `Downloading ${label} (${loadedMb} MB)`, null);
    }
  }

  const data = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    data.set(chunk, offset);
    offset += chunk.byteLength;
  }

  report(onProgress, `Downloaded ${label}`, progressEnd);
  await cache?.put(url, new Response(data, {
    headers: {
      'Content-Type': response.headers.get('Content-Type') || 'application/octet-stream',
      'Content-Length': String(data.byteLength),
    },
  }));
  return data;
}

async function loadSession(
  baseUrl: string,
  name: string,
  provider: 'webgpu' | 'wasm',
  progressStart: number,
  progressEnd: number,
  onProgress?: AsrProgress,
) {
  const graphName = `${name}_${MODEL_DTYPES.asr}`;
  const onnxPath = `${baseUrl}/onnx/${graphName}.onnx`;
  const dataPath = `${baseUrl}/onnx/${graphName}.onnx_data`;
  const midProgress = progressStart + (progressEnd - progressStart) * 0.25;
  const dataProgress = progressStart + (progressEnd - progressStart) * 0.85;

  const onnxBytes = await fetchBinaryWithProgress(
    onnxPath,
    `${graphName}.onnx`,
    progressStart,
    midProgress,
    onProgress,
  );
  const externalBytes = await fetchBinaryWithProgress(
    dataPath,
    `${graphName}.onnx_data`,
    midProgress,
    dataProgress,
    onProgress,
  );

  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: provider === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    externalData: [
      {
        path: `${graphName}.onnx_data`,
        data: externalBytes,
      },
    ],
  };

  report(onProgress, `Creating ${graphName} session on ${provider}`, dataProgress);
  try {
    const session = await ort.InferenceSession.create(onnxBytes, options);
    report(onProgress, `${graphName} session ready`, progressEnd);
    return session;
  } catch (error) {
    if (!isOrtSessionCreationRace(error)) throw error;
    report(onProgress, `Waiting for ONNX Runtime session slot for ${graphName}`, null);
    await sleep(750);
    const session = await ort.InferenceSession.create(onnxBytes, options);
    report(onProgress, `${graphName} session ready`, progressEnd);
    return session;
  }
}

async function loadEmbedTokens(baseUrl: string, onProgress?: AsrProgress) {
  report(onProgress, 'Loading ASR text embedding metadata', 35);
  const metaResponse = await fetch(`${baseUrl}/onnx/embed_tokens.json`, { credentials: 'omit' });

  if (!metaResponse.ok) {
    throw new Error(`Failed to load embed_tokens.json: ${metaResponse.status}`);
  }

  const meta = await metaResponse.json();
  const embedBytes = await fetchBinaryWithProgress(
    `${baseUrl}/onnx/embed_tokens.bin`,
    'embed_tokens.bin',
    38,
    70,
    onProgress,
  );
  const hiddenSize = Number(meta.hidden_size ?? meta.hiddenSize ?? meta.shape?.[1]);
  const vocabSize = Number(meta.vocab_size ?? meta.vocabSize ?? meta.shape?.[0]);

  if (!Number.isFinite(hiddenSize) || !Number.isFinite(vocabSize)) {
    throw new Error('embed_tokens metadata is missing shape information');
  }
  if (embedBytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error('embed_tokens.bin byte length is not aligned to float32 values');
  }

  return {
    weight: new Float32Array(
      embedBytes.buffer,
      embedBytes.byteOffset,
      embedBytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
    ),
    shape: { vocabSize, hiddenSize },
  };
}

export async function loadAsrSession(onProgress?: AsrProgress): Promise<AsrSession> {
  if (asrPromise) return asrPromise;

  asrPromise = (async () => {
    const baseUrl = modelBaseUrl();
    const browserNavigator = navigator as Navigator & { gpu?: unknown };
    const backend: 'webgpu' | 'wasm' = browserNavigator.gpu ? 'webgpu' : 'wasm';

    try {
      ort.env.wasm.numThreads = 1;
      report(onProgress, 'Loading LFM2.5-Audio tokenizer', 0);
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_IDS.asr);

      report(onProgress, 'Loading LFM2.5-Audio config', 5);
      const configResponse = await fetch(`${baseUrl}/config.json`, { credentials: 'omit' });
      if (!configResponse.ok) {
        throw new Error(`Failed to load config.json: ${configResponse.status}`);
      }
      const config = await configResponse.json();
      const lfmConfig = config.lfm || {};

      report(onProgress, `Loading LFM2.5-Audio audio encoder on ${backend}`, 10);
      const audioEncoder = await loadSession(baseUrl, 'audio_encoder', backend, 10, 35, onProgress);

      const embed = await loadEmbedTokens(baseUrl, onProgress);

      report(onProgress, `Loading LFM2.5-Audio decoder on ${backend}`, 75);
      const decoder = await loadSession(baseUrl, 'decoder', backend, 75, 98, onProgress);

      onProgress?.({
        status: 'ready',
        model: 'asr',
        message: `LFM2.5-Audio ready on ${backend}`,
        progress: 100,
      });

      return {
        tokenizer,
        decoder,
        audioEncoder,
        embedTokens: embed.weight,
        embedShape: embed.shape,
        modelBaseUrl: baseUrl,
        backend,
        layerTypes: lfmConfig.layer_types || [],
        hiddenSize: lfmConfig.hidden_size || 2048,
        numKVHeads: lfmConfig.num_key_value_heads || 8,
        headDim: Math.floor((lfmConfig.hidden_size || 2048) / (lfmConfig.num_attention_heads || 32)),
        convL: lfmConfig.conv_L_cache || 3,
        vocabSize: lfmConfig.vocab_size || 65536,
      };
    } catch (error) {
      asrPromise = null;
      onProgress?.({
        status: 'error',
        model: 'asr',
        message: error instanceof Error ? error.message : String(error),
        progress: null,
      });
      throw error;
    }
  })();

  return asrPromise;
}

export function resetAsrSessionForTests() {
  asrPromise = null;
}

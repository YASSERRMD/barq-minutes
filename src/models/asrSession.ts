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

async function loadSession(
  baseUrl: string,
  name: string,
  provider: 'webgpu' | 'wasm',
  onProgress?: AsrProgress,
) {
  const graphName = `${name}_${MODEL_DTYPES.asr}`;
  const onnxPath = `${baseUrl}/onnx/${graphName}.onnx`;
  report(onProgress, `Loading ${graphName}.onnx`, null);

  const externalData = [
    {
      path: `${graphName}.onnx_data`,
      data: `${baseUrl}/onnx/${graphName}.onnx_data`,
    },
  ];

  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: provider === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    externalData,
  };

  try {
    return await ort.InferenceSession.create(onnxPath, options);
  } catch (error) {
    if (!isOrtSessionCreationRace(error)) throw error;
    report(onProgress, `Waiting for ONNX Runtime session slot for ${graphName}`, null);
    await sleep(750);
    return ort.InferenceSession.create(onnxPath, options);
  }
}

async function loadEmbedTokens(baseUrl: string, onProgress?: AsrProgress) {
  report(onProgress, 'Loading ASR text embeddings', 70);
  const [metaResponse, binResponse] = await Promise.all([
    fetch(`${baseUrl}/onnx/embed_tokens.json`, { credentials: 'omit' }),
    fetch(`${baseUrl}/onnx/embed_tokens.bin`, { credentials: 'omit' }),
  ]);

  if (!metaResponse.ok) {
    throw new Error(`Failed to load embed_tokens.json: ${metaResponse.status}`);
  }
  if (!binResponse.ok) {
    throw new Error(`Failed to load embed_tokens.bin: ${binResponse.status}`);
  }

  const meta = await metaResponse.json();
  const buffer = await binResponse.arrayBuffer();
  const hiddenSize = Number(meta.hidden_size ?? meta.hiddenSize ?? meta.shape?.[1]);
  const vocabSize = Number(meta.vocab_size ?? meta.vocabSize ?? meta.shape?.[0]);

  if (!Number.isFinite(hiddenSize) || !Number.isFinite(vocabSize)) {
    throw new Error('embed_tokens metadata is missing shape information');
  }

  return {
    weight: new Float32Array(buffer),
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

      report(onProgress, `Loading LFM2.5-Audio decoder on ${backend}`, 20);
      const decoder = await loadSession(baseUrl, 'decoder', backend, onProgress);

      report(onProgress, `Loading LFM2.5-Audio audio encoder on ${backend}`, 50);
      const audioEncoder = await loadSession(baseUrl, 'audio_encoder', backend, onProgress);

      const embed = await loadEmbedTokens(baseUrl, onProgress);

      report(onProgress, 'Loading LFM2.5-Audio config', 80);
      const configResponse = await fetch(`${baseUrl}/config.json`, { credentials: 'omit' });
      if (!configResponse.ok) {
        throw new Error(`Failed to load config.json: ${configResponse.status}`);
      }
      const config = await configResponse.json();
      const lfmConfig = config.lfm || {};

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

import {
  AutoProcessor,
  AutoTokenizer,
  TextStreamer,
  WhisperForConditionalGeneration,
  env,
  full,
} from '@huggingface/transformers';
import { MODEL_IDS, type ModelLoadProgress } from './modelConfig';

type AsrProgress = (progress: ModelLoadProgress) => void;

export type AsrSession = {
  tokenizer: any;
  processor: any;
  model: any;
  backend: 'webgpu' | 'wasm';
  transcribe(audio: Float32Array, options?: {
    language?: string;
    maxNewTokens?: number;
    onToken?: (text: string) => void;
  }): Promise<string>;
};

let asrPromise: Promise<AsrSession> | null = null;

env.allowLocalModels = false;
env.useBrowserCache = true;
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
}

function report(
  onProgress: AsrProgress | undefined,
  message: string,
  progress: number | null,
  status: ModelLoadProgress['status'] = 'loading',
) {
  onProgress?.({
    status,
    model: 'asr',
    message,
    progress,
  });
}

function isOrtSessionCreationRace(error: unknown) {
  const message = String(error);
  return message.includes('another WebGPU EP inference session is being created')
    || message.includes("multiple calls to 'initWasm()' detected")
    || message.includes('no available backend found');
}

function progressReporter(onProgress: AsrProgress | undefined) {
  return (event: any) => {
    if (event.status === 'progress') {
      report(onProgress, `Loading ${event.file ?? 'Whisper model'} from browser cache`, event.progress ?? null);
      return;
    }
    if (event.status === 'ready') {
      report(onProgress, `Loaded ${event.file ?? 'Whisper file'}`, null);
    }
  };
}

function cleanTranscript(text: string) {
  return text
    .replace(/\s+/g, ' ')
    .trim();
}

async function loadWhisperModel(onProgress?: AsrProgress) {
  const common = {
    progress_callback: progressReporter(onProgress),
  };

  report(onProgress, 'Loading Whisper tokenizer', 0);
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_IDS.asr, common);

  report(onProgress, 'Loading Whisper processor', 10);
  const processor = await AutoProcessor.from_pretrained(MODEL_IDS.asr, common);

  const browserNavigator = navigator as Navigator & { gpu?: unknown };
  const hasWebGpu = Boolean(browserNavigator.gpu);
  let backend: 'webgpu' | 'wasm' = hasWebGpu ? 'webgpu' : 'wasm';

  report(onProgress, `Loading Whisper ASR on ${backend}`, 20);
  let model: any;
  try {
    model = await WhisperForConditionalGeneration.from_pretrained(MODEL_IDS.asr, {
      dtype: {
        encoder_model: 'fp32',
        decoder_model_merged: 'q4',
      },
      device: backend,
      progress_callback: progressReporter(onProgress),
    } as any);
  } catch (error) {
    if (!hasWebGpu || !isOrtSessionCreationRace(error)) throw error;
    report(onProgress, 'Retrying Whisper ASR after WebGPU session slot wait', null);
    await new Promise((resolve) => window.setTimeout(resolve, 750));
    model = await WhisperForConditionalGeneration.from_pretrained(MODEL_IDS.asr, {
      dtype: {
        encoder_model: 'fp32',
        decoder_model_merged: 'q4',
      },
      device: backend,
      progress_callback: progressReporter(onProgress),
    } as any);
  }

  if (backend === 'webgpu') {
    report(onProgress, 'Compiling Whisper WebGPU shaders', 88);
    await model.generate({
      input_features: full([1, 80, 3000], 0),
      max_new_tokens: 1,
    });
  }

  report(onProgress, `Whisper ready on ${backend}`, 100, 'ready');

  return { tokenizer, processor, model, backend };
}

export async function loadAsrSession(onProgress?: AsrProgress): Promise<AsrSession> {
  if (asrPromise) return asrPromise;

  asrPromise = (async () => {
    try {
      const { tokenizer, processor, model, backend } = await loadWhisperModel(onProgress);

      return {
        tokenizer,
        processor,
        model,
        backend,
        async transcribe(audio, options = {}) {
          let currentText = '';
          const streamer = new TextStreamer(tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (token: string) => {
              currentText = cleanTranscript(`${currentText} ${token}`);
              options.onToken?.(currentText);
            },
          });

          const inputs = await processor(audio);
          const output = await model.generate({
            ...inputs,
            max_new_tokens: options.maxNewTokens ?? 128,
            language: options.language ?? 'en',
            streamer,
          });
          const decoded = tokenizer.batch_decode(output, { skip_special_tokens: true });
          return cleanTranscript(Array.isArray(decoded) ? decoded[0] ?? currentText : String(decoded));
        },
      };
    } catch (error) {
      asrPromise = null;
      report(onProgress, error instanceof Error ? error.message : String(error), null, 'error');
      throw error;
    }
  })();

  return asrPromise;
}

export function resetAsrSessionForTests() {
  asrPromise = null;
}

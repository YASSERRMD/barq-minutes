import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  env,
} from '@huggingface/transformers';
import {
  LLM_INPUT_BUDGET_TOKENS,
  MODEL_DTYPES,
  MODEL_IDS,
  type ModelLoadProgress,
} from './modelConfig';
import { estimateTokens } from '../utils/tokens';

type ChatRole = 'system' | 'user' | 'assistant';

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type GenerateOptions = {
  maxNewTokens: number;
  temperature: number;
  onToken?: (token: string) => void;
  onProgress?: (progress: ModelLoadProgress) => void;
};

export type LlmSession = {
  generate(messages: ChatMessage[], options: GenerateOptions): Promise<string>;
  tokenizer: unknown;
  model: unknown;
};

let sessionPromise: Promise<LlmSession> | null = null;

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isOrtSessionCreationRace(error: unknown) {
  const message = String(error);
  return message.includes('another WebGPU EP inference session is being created')
    || message.includes("multiple calls to 'initWasm()' detected")
    || message.includes('no available backend found');
}

env.allowLocalModels = false;
env.useBrowserCache = true;
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
}

function report(
  onProgress: GenerateOptions['onProgress'],
  message: string,
  progress: number | null,
  status: ModelLoadProgress['status'] = 'loading',
) {
  onProgress?.({
    status,
    model: 'llm',
    message,
    progress,
  });
}

function cleanGeneratedText(text: string): string {
  return text
    .replace(/<\|im_end\|>/g, '')
    .replace(/<\|im_start\|>assistant/g, '')
    .trim();
}

function assertPromptBudget(messages: ChatMessage[]) {
  const roughTokens = estimateTokens(messages.map((message) => message.content).join('\n'));
  if (roughTokens > LLM_INPUT_BUDGET_TOKENS) {
    throw new Error(
      `Prompt is ${roughTokens} estimated tokens. Limit is ${LLM_INPUT_BUDGET_TOKENS}.`,
    );
  }
}

export async function loadLlmSession(
  onProgress?: GenerateOptions['onProgress'],
): Promise<LlmSession> {
  if (sessionPromise) return sessionPromise;

  sessionPromise = (async () => {
    report(onProgress, 'Loading glm5.1-distill tokenizer', 0);
    try {
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_IDS.llm, {
        progress_callback: (event: any) => {
          if (event.status === 'progress') {
            report(onProgress, `Loading ${event.file ?? 'tokenizer'} from model cache`, event.progress ?? null);
          }
        },
      });

      report(onProgress, 'Loading glm5.1-distill Q4 model on WebGPU', 20);
      let model: any;
      try {
        const modelOptions = {
          device: 'webgpu',
          dtype: MODEL_DTYPES.llm,
          progress_callback: (event: any) => {
            if (event.status === 'progress') {
              report(onProgress, `Loading ${event.file ?? 'model weights'} from model cache`, event.progress ?? null);
            }
          },
        } as const;
        try {
          model = await AutoModelForCausalLM.from_pretrained(MODEL_IDS.llm, modelOptions);
        } catch (error) {
          if (!isOrtSessionCreationRace(error)) throw error;
          report(onProgress, 'Waiting for ONNX Runtime session slot for glm5.1-distill', null);
          await sleep(750);
          model = await AutoModelForCausalLM.from_pretrained(MODEL_IDS.llm, modelOptions);
        }
      } catch (webgpuError) {
        report(onProgress, 'WebGPU load failed, using WASM fallback', 45);
        console.warn('[llmSession] WebGPU load failed, falling back to WASM', webgpuError);
        model = await AutoModelForCausalLM.from_pretrained(MODEL_IDS.llm, {
          device: 'wasm',
          dtype: MODEL_DTYPES.llm,
        });
      }

      report(onProgress, 'glm5.1-distill ready', 100, 'ready');

      return {
        tokenizer,
        model,
        async generate(messages: ChatMessage[], options: GenerateOptions) {
          assertPromptBudget(messages);

          const input = (tokenizer as any).apply_chat_template(messages, {
            add_generation_prompt: true,
            return_dict: true,
          });

          let streamed = '';
          const streamer = new TextStreamer(tokenizer as any, {
            skip_prompt: true,
            skip_special_tokens: false,
            callback_function: (token: string) => {
              streamed += token;
              options.onToken?.(token);
            },
          });

          const output = await model.generate({
            ...input,
            max_new_tokens: options.maxNewTokens,
            temperature: options.temperature,
            do_sample: options.temperature > 0,
            top_k: 50,
            top_p: options.temperature <= 0.2 ? 0.1 : 0.9,
            repetition_penalty: 1.05,
            streamer,
          });

          if (streamed) return cleanGeneratedText(streamed);
          const decoded = (tokenizer as any).decode(output[0], {
            skip_special_tokens: true,
          });
          return cleanGeneratedText(decoded);
        },
      };
    } catch (error) {
      sessionPromise = null;
      report(onProgress, error instanceof Error ? error.message : String(error), null, 'error');
      throw error;
    }
  })();

  return sessionPromise;
}

export function resetLlmSessionForTests() {
  sessionPromise = null;
}

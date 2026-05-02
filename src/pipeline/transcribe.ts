import * as ort from 'onnxruntime-web/webgpu';
import type { AsrSession } from '../models/asrSession';
import type { TranscriptTurn } from '../schemas/meeting';
import { computeMelSpectrogram } from '../utils/audioProcessor';

const SPECIAL_TOKENS = { IM_END: 7 };

export type TranscriptionProgress = (event: {
  step: 'decode_audio' | 'transcribe';
  message: string;
  progress: number | null;
}) => void;

async function decodeBlob(blob: Blob): Promise<AudioBuffer> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    return await audioContext.decodeAudioData(arrayBuffer);
  } finally {
    await audioContext.close();
  }
}

function downmix(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0).slice();
  const length = buffer.length;
  const samples = new Float32Array(length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < length; i++) {
      samples[i] += data[i] / buffer.numberOfChannels;
    }
  }
  return samples;
}

function toNumberTokenId(value: unknown): number {
  const tokenId = typeof value === 'bigint' ? Number(value) : Number(value);
  if (!Number.isSafeInteger(tokenId) || tokenId < 0) {
    throw new Error(`Invalid tokenizer id: ${String(value)}`);
  }
  return tokenId;
}

function encodeTokenIds(tokenizer: any, text: string): number[] {
  const encoded = tokenizer(text, { add_special_tokens: false });
  const data = encoded?.input_ids?.data ?? encoded?.input_ids ?? encoded;
  return Array.from(data, toNumberTokenId);
}

function getTokenizerEosId(tokenizer: any): number | null {
  if (tokenizer.eos_token_id === null || tokenizer.eos_token_id === undefined) return null;
  return toNumberTokenId(tokenizer.eos_token_id);
}

/**
 * Transcribes a recorded audio blob using the pre-loaded ASR session from the
 * model boot context. The session is passed in, never lazy-loaded here, so
 * model downloads cannot block post-recording processing.
 *
 * @param blob Raw audio blob from MediaRecorder.
 * @param asrSession Pre-loaded LFM2.5-Audio ONNX session from ModelBootContext.
 * @param options Meeting title, optional fallback text, and progress callback.
 *
 * Replaces the fallback stub with real ONNX inference using `audioEncoder` / `decoder`
 * and LFM2.5-Audio-1.5B prompt structures.
 */
export async function transcribeAudioBlob(
  blob: Blob,
  asrSession: AsrSession,
  options: {
    meetingTitle: string;
    fallbackText?: string;
    onProgress?: TranscriptionProgress;
  },
): Promise<TranscriptTurn[]> {
  options.onProgress?.({
    step: 'decode_audio',
    message: 'Decoding recorded audio locally',
    progress: 20,
  });

  const audio = await decodeBlob(blob);
  const _samples = downmix(audio);
  const durationSec = _samples.length / audio.sampleRate;

  options.onProgress?.({
    step: 'transcribe',
    message: 'Running local ASR inference',
    progress: 50,
  });

  const { melFeatures, numFrames } = computeMelSpectrogram(_samples, audio.sampleRate);

  const melTensor = new ort.Tensor('float32', melFeatures, [1, numFrames, 128]);
  const melLengths = new ort.Tensor('int64', new BigInt64Array([BigInt(numFrames)]), [1]);
  const encoderOut = await asrSession.audioEncoder.run({ mel_spectrogram: melTensor, mel_lengths: melLengths });
  const audioEmbeds = encoderOut.audio_embeddings;
  const audioLen = audioEmbeds.dims[1];

  const prefixText = `<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n<|im_start|>user\n`;
  const suffixText = `<|im_end|>\n<|im_start|>assistant\n`;

  const prefixIds = encodeTokenIds(asrSession.tokenizer, prefixText);
  const suffixIds = encodeTokenIds(asrSession.tokenizer, suffixText);
  const prefixLen = prefixIds.length;
  const suffixLen = suffixIds.length;
  const totalLen = prefixLen + audioLen + suffixLen;

  const getTextEmbeddings = (tokenIds: number[]) => {
    const seqLen = tokenIds.length;
    const embeddings = new Float32Array(seqLen * asrSession.hiddenSize);
    for (let i = 0; i < seqLen; i++) {
      const tokenId = toNumberTokenId(tokenIds[i]);
      const srcOffset = tokenId * asrSession.hiddenSize;
      const srcEnd = srcOffset + asrSession.hiddenSize;
      if (srcEnd > asrSession.embedTokens.length) {
        throw new Error(`Token id ${tokenId} is outside the ASR embedding table`);
      }
      embeddings.set(asrSession.embedTokens.subarray(srcOffset, srcOffset + asrSession.hiddenSize), i * asrSession.hiddenSize);
    }
    return embeddings;
  };

  const allEmbeds = new Float32Array(totalLen * asrSession.hiddenSize);
  allEmbeds.set(getTextEmbeddings(prefixIds), 0);
  const audioEmbedsData = audioEmbeds.data as Float32Array;
  allEmbeds.set(new Float32Array(audioEmbedsData.buffer, audioEmbedsData.byteOffset, audioLen * asrSession.hiddenSize), prefixLen * asrSession.hiddenSize);
  allEmbeds.set(getTextEmbeddings(suffixIds), (prefixLen + audioLen) * asrSession.hiddenSize);

  const cache: Record<string, ort.Tensor> = {};
  for (let idx = 0; idx < asrSession.layerTypes.length; idx++) {
    if (asrSession.layerTypes[idx] === 'conv') {
      cache[`past_conv.${idx}`] = new ort.Tensor('float32', new Float32Array(1 * asrSession.hiddenSize * asrSession.convL), [1, asrSession.hiddenSize, asrSession.convL]);
    } else {
      cache[`past_key_values.${idx}.key`] = new ort.Tensor('float32', new Float32Array(0), [1, asrSession.numKVHeads, 0, asrSession.headDim]);
      cache[`past_key_values.${idx}.value`] = new ort.Tensor('float32', new Float32Array(0), [1, asrSession.numKVHeads, 0, asrSession.headDim]);
    }
  }

  let inputEmbeds = new ort.Tensor('float32', allEmbeds, [1, totalLen, asrSession.hiddenSize]);
  let attentionMask = new ort.Tensor('int64', new BigInt64Array(totalLen).fill(1n), [1, totalLen]);

  let decoderOut = await asrSession.decoder.run({ inputs_embeds: inputEmbeds, attention_mask: attentionMask, ...cache });

  const updateCache = (outputs: Record<string, ort.Tensor>) => {
    for (const name of Object.keys(outputs)) {
      if (name.startsWith('present_conv.')) cache[name.replace('present_conv', 'past_conv')] = outputs[name];
      else if (name.startsWith('present.')) cache[name.replace('present.', 'past_key_values.')] = outputs[name];
    }
  };

  updateCache(decoderOut);

  const generatedTokens: number[] = [];
  let currentLen = totalLen;
  const maxTokens = 100;
  const eosTokenId = getTokenizerEosId(asrSession.tokenizer);

  for (let i = 0; i < maxTokens; i++) {
    const logitsData = decoderOut.logits.data as Float32Array;
    const seqLen = decoderOut.logits.dims[1];
    const offset = (seqLen - 1) * asrSession.vocabSize;
    let maxIdx = 0;
    let maxVal = logitsData[offset];
    for (let j = 1; j < asrSession.vocabSize; j++) {
      if (logitsData[offset + j] > maxVal) { maxVal = logitsData[offset + j]; maxIdx = j; }
    }

    if (maxIdx === eosTokenId || maxIdx === SPECIAL_TOKENS.IM_END) {
      break;
    }
    generatedTokens.push(maxIdx);

    inputEmbeds = new ort.Tensor('float32', getTextEmbeddings([maxIdx]), [1, 1, asrSession.hiddenSize]);
    currentLen++;
    attentionMask = new ort.Tensor('int64', new BigInt64Array(currentLen).fill(1n), [1, currentLen]);
    decoderOut = await asrSession.decoder.run({ inputs_embeds: inputEmbeds, attention_mask: attentionMask, ...cache });
    updateCache(decoderOut);
  }

  const generatedText = (asrSession.tokenizer as any).decode(generatedTokens, { skip_special_tokens: true });

  const text = generatedText.trim() || options.fallbackText?.trim() || `[Audio recorded for ${options.meetingTitle}]`;

  options.onProgress?.({
    step: 'transcribe',
    message: 'Transcript ready',
    progress: 100,
  });

  return [
    {
      speaker: 'Speaker 1',
      text,
      startSec: 0,
      endSec: Math.max(1, durationSec),
    },
  ];
}

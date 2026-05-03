import type { AsrSession } from '../models/asrSession';
import type { TranscriptTurn } from '../schemas/meeting';

export const ASR_SAMPLE_RATE = 16_000;
const WHISPER_CHUNK_SEC = 25;
const MIN_CHUNK_SEC = 0.25;

export type TranscriptionProgress = (event: {
  step: 'decode_audio' | 'transcribe';
  message: string;
  progress: number | null;
}) => void;

async function decodeBlob(blob: Blob): Promise<AudioBuffer> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext({ sampleRate: 16000 });
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

function resampleLinear(samples: Float32Array, sourceRate: number, targetRate: number) {
  if (sourceRate === targetRate) return samples;
  const ratio = sourceRate / targetRate;
  const nextLength = Math.max(1, Math.round(samples.length / ratio));
  const resampled = new Float32Array(nextLength);

  for (let i = 0; i < nextLength; i++) {
    const sourceIndex = i * ratio;
    const left = Math.floor(sourceIndex);
    const right = Math.min(samples.length - 1, left + 1);
    const weight = sourceIndex - left;
    resampled[i] = samples[left] * (1 - weight) + samples[right] * weight;
  }

  return resampled;
}

export async function decodeAudioBlobToSamples(blob: Blob): Promise<{
  samples: Float32Array;
  sampleRate: number;
  durationSec: number;
}> {
  const audio = await decodeBlob(blob);
  const samples = resampleLinear(downmix(audio), audio.sampleRate, ASR_SAMPLE_RATE);
  return {
    samples,
    sampleRate: ASR_SAMPLE_RATE,
    durationSec: samples.length / ASR_SAMPLE_RATE,
  };
}

export async function transcribeAudioSamples(
  samples: Float32Array,
  sampleRate: number,
  asrSession: AsrSession,
  options: {
    meetingTitle: string;
    fallbackText?: string;
    allowEmpty?: boolean;
    startSec?: number;
    maxTokensPerChunk?: number;
    onProgress?: TranscriptionProgress;
  },
): Promise<TranscriptTurn[]> {
  const normalized = resampleLinear(samples, sampleRate, ASR_SAMPLE_RATE);
  const chunkSize = WHISPER_CHUNK_SEC * ASR_SAMPLE_RATE;
  const totalChunks = Math.max(1, Math.ceil(normalized.length / chunkSize));
  const turns: TranscriptTurn[] = [];

  for (let index = 0; index < totalChunks; index++) {
    const startSample = index * chunkSize;
    const endSample = Math.min(normalized.length, startSample + chunkSize);
    if (endSample - startSample < MIN_CHUNK_SEC * ASR_SAMPLE_RATE) continue;

    const chunk = normalized.slice(startSample, endSample);
    const chunkStartSec = (options.startSec ?? 0) + startSample / ASR_SAMPLE_RATE;
    const chunkEndSec = (options.startSec ?? 0) + endSample / ASR_SAMPLE_RATE;
    const chunkDurationSec = chunk.length / ASR_SAMPLE_RATE;
    const progress = ((index + 0.5) / totalChunks) * 100;

    options.onProgress?.({
      step: 'transcribe',
      message: totalChunks === 1
        ? `Running Whisper ASR on ${asrSession.backend}`
        : `Running Whisper ASR on chunk ${index + 1}/${totalChunks}`,
      progress,
    });

    const generatedText = await asrSession.transcribe(chunk, {
      maxNewTokens: options.maxTokensPerChunk ?? Math.min(224, Math.max(96, Math.ceil(chunkDurationSec * 8))),
      language: 'en',
      onToken: (text) => {
        options.onProgress?.({
          step: 'transcribe',
          message: `Transcribing locally: ${text.slice(-48)}`,
          progress: null,
        });
      },
    });

    const text = generatedText.trim();
    if (text) {
      turns.push({
        speaker: 'Speaker 1',
        text,
        startSec: chunkStartSec,
        endSec: Math.max(chunkStartSec + MIN_CHUNK_SEC, chunkEndSec),
      });
    }
  }

  if (turns.length === 0 && options.allowEmpty) {
    return [
      {
        speaker: 'Speaker 1',
        text: '',
        startSec: options.startSec ?? 0,
        endSec: (options.startSec ?? 0) + Math.max(1, normalized.length / ASR_SAMPLE_RATE),
      },
    ];
  }

  if (turns.length === 0 && options.fallbackText?.trim()) {
    return [
      {
        speaker: 'Speaker 1',
        text: options.fallbackText.trim(),
        startSec: options.startSec ?? 0,
        endSec: (options.startSec ?? 0) + Math.max(1, normalized.length / ASR_SAMPLE_RATE),
      },
    ];
  }

  if (turns.length === 0) {
    return [
      {
        speaker: 'Speaker 1',
        text: `[Audio recorded for ${options.meetingTitle}]`,
        startSec: options.startSec ?? 0,
        endSec: (options.startSec ?? 0) + Math.max(1, normalized.length / ASR_SAMPLE_RATE),
      },
    ];
  }

  return turns;
}

export async function transcribeAudioBlob(
  blob: Blob,
  asrSession: AsrSession,
  options: {
    meetingTitle: string;
    fallbackText?: string;
    allowEmpty?: boolean;
    maxTokens?: number;
    onProgress?: TranscriptionProgress;
  },
): Promise<TranscriptTurn[]> {
  options.onProgress?.({
    step: 'decode_audio',
    message: 'Decoding audio locally',
    progress: 20,
  });

  const decoded = await decodeAudioBlobToSamples(blob);
  const turns = await transcribeAudioSamples(decoded.samples, decoded.sampleRate, asrSession, {
    meetingTitle: options.meetingTitle,
    fallbackText: options.fallbackText,
    allowEmpty: options.allowEmpty,
    maxTokensPerChunk: options.maxTokens,
    onProgress: options.onProgress,
  });

  options.onProgress?.({
    step: 'transcribe',
    message: 'Transcript ready',
    progress: 100,
  });

  return turns;
}

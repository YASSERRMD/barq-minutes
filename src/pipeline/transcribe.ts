import type { AsrSession } from '../models/asrSession';
import type { TranscriptTurn } from '../schemas/meeting';

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

  const audio = await decodeBlob(blob);
  const samples = downmix(audio);
  const durationSec = samples.length / audio.sampleRate;

  options.onProgress?.({
    step: 'transcribe',
    message: `Running Whisper ASR on ${asrSession.backend}`,
    progress: 50,
  });

  const generatedText = await asrSession.transcribe(samples, {
    maxNewTokens: options.maxTokens ?? Math.min(448, Math.max(96, Math.ceil(durationSec * 4))),
    language: 'english',
    onToken: (text) => {
      options.onProgress?.({
        step: 'transcribe',
        message: `Transcribing locally: ${text.slice(-48)}`,
        progress: null,
      });
    },
  });
  const text = generatedText.trim();

  options.onProgress?.({
    step: 'transcribe',
    message: 'Transcript ready',
    progress: 100,
  });

  if (!text && options.allowEmpty) {
    return [
      {
        speaker: 'Speaker 1',
        text: '',
        startSec: 0,
        endSec: Math.max(1, durationSec),
      },
    ];
  }

  return [
    {
      speaker: 'Speaker 1',
      text: text || options.fallbackText?.trim() || `[Audio recorded for ${options.meetingTitle}]`,
      startSec: 0,
      endSec: Math.max(1, durationSec),
    },
  ];
}

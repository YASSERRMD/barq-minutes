import type { AsrSession } from '../models/asrSession';
import type { TranscriptTurn } from '../schemas/meeting';

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

/**
 * Transcribes a recorded audio blob using the pre-loaded ASR session from the
 * model boot context.  The session is passed in — never lazy-loaded here — so
 * model downloads cannot block post-recording processing.
 *
 * @param blob       - Raw audio blob from MediaRecorder.
 * @param asrSession - Pre-loaded LFM2.5-Audio ONNX session (from ModelBootContext).
 * @param options    - Meeting title, optional fallback text, and progress callback.
 *
 * TODO: Replace the fallback stub below with real ONNX inference once the exact
 *       input tensor shapes for `audioEncoder` / `decoder` are confirmed from
 *       the LiquidAI/LFM2.5-Audio-1.5B-ONNX model card.
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
  const _samples = downmix(audio); // passed to encoder once real inference is wired
  const durationSec = _samples.length / audio.sampleRate;

  options.onProgress?.({
    step: 'transcribe',
    message: 'Running local ASR inference',
    progress: 50,
  });

  // --- Stub: real ONNX decode path ---
  // When confirmed, replace this block with:
  //   const encoderOut = await asrSession.audioEncoder.run({ audio_pcm: new ort.Tensor('float32', _samples, [1, _samples.length]) });
  //   const decoderOut = await asrSession.decoder.run({ encoder_hidden_states: encoderOut.last_hidden_state, ... });
  //   const text = (asrSession.tokenizer as any).decode(decoderOut.sequences[0], { skip_special_tokens: true });
  void asrSession; // suppress unused-variable warning until inference is wired

  const text =
    options.fallbackText?.trim() ||
    `Audio recorded locally for ${options.meetingTitle}. ` +
      `ASR session loaded (backend: ${asrSession.backend}) — inference stub active.`;

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

export const DEFAULT_MEL_CONFIG = {
  sample_rate: 16000,
  n_fft: 512,
  win_length: 400,
  hop_length: 160,
  n_mels: 128,
  fmin: 0,
  fmax: 8000,
  preemph: 0.97,
  log_zero_guard: 5.960464477539063e-08,
  normalize: 'per_feature',
  mel_norm: 'slaney',
};

let melFilterbank: Float32Array[] | null = null;

function createMelFilterbank(sr: number, nFft: number, nMels: number, fmin: number, fmax: number) {
  const nFreqs = Math.floor(nFft / 2) + 1;
  const hzToMel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
  const melToHz = (mel: number) => 700 * (Math.pow(10, mel / 2595) - 1);

  const melMin = hzToMel(fmin);
  const melMax = hzToMel(fmax);
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
  }

  const hzPoints = Array.from(melPoints).map(melToHz);
  const binPoints = hzPoints.map((hz) => Math.floor((nFft + 1) * hz / sr));

  const filterbank: Float32Array[] = [];
  for (let m = 0; m < nMels; m++) {
    const filter = new Float32Array(nFreqs);
    const start = binPoints[m];
    const center = binPoints[m + 1];
    const end = binPoints[m + 2];

    for (let k = start; k < center; k++) {
      if (k < nFreqs) filter[k] = (k - start) / (center - start);
    }
    for (let k = center; k < end; k++) {
      if (k < nFreqs) filter[k] = (end - k) / (end - center);
    }

    const enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);
    for (let k = 0; k < nFreqs; k++) filter[k] *= enorm;
    filterbank.push(filter);
  }
  return filterbank;
}

function createHannWindow(length: number) {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return window;
}

function resampleAudio(audio: Float32Array, srcSr: number, dstSr: number) {
  if (srcSr === dstSr) return audio;
  const ratio = srcSr / dstSr;
  const newLength = Math.floor(audio.length / ratio);
  const resampled = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const srcIdx = i * ratio;
    const srcIdxFloor = Math.floor(srcIdx);
    const srcIdxCeil = Math.min(srcIdxFloor + 1, audio.length - 1);
    const frac = srcIdx - srcIdxFloor;
    resampled[i] = audio[srcIdxFloor] * (1 - frac) + audio[srcIdxCeil] * frac;
  }
  return resampled;
}

type FFTCache = {
  n: number;
  twiddleRe: Float32Array;
  twiddleIm: Float32Array;
  bitrev: Uint32Array;
  workRe: Float32Array;
  workIm: Float32Array;
};
let _fftCache: FFTCache | null = null;

function initFFT(n: number) {
  if (_fftCache && _fftCache.n === n) return _fftCache;
  const twiddleRe = new Float32Array(n / 2);
  const twiddleIm = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    const angle = (-2 * Math.PI * i) / n;
    twiddleRe[i] = Math.cos(angle);
    twiddleIm[i] = Math.sin(angle);
  }
  const bitrev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let j = 0;
    let x = i;
    for (let k = 1; k < n; k <<= 1) {
      j = (j << 1) | (x & 1);
      x >>= 1;
    }
    bitrev[i] = j;
  }
  _fftCache = { n, twiddleRe, twiddleIm, bitrev, workRe: new Float32Array(n), workIm: new Float32Array(n) };
  return _fftCache;
}

function computeRfftMagnitude(frame: Float32Array) {
  const n = frame.length;
  const nFreqs = Math.floor(n / 2) + 1;
  const cache = initFFT(n);
  const { twiddleRe, twiddleIm, bitrev, workRe, workIm } = cache;

  for (let i = 0; i < n; i++) {
    workRe[bitrev[i]] = frame[i];
    workIm[bitrev[i]] = 0;
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < halfLen; j++) {
        const twIdx = j * step;
        const wRe = twiddleRe[twIdx];
        const wIm = twiddleIm[twIdx];
        const u = i + j;
        const v = u + halfLen;
        const tRe = wRe * workRe[v] - wIm * workIm[v];
        const tIm = wRe * workIm[v] + wIm * workRe[v];
        workRe[v] = workRe[u] - tRe;
        workIm[v] = workIm[u] - tIm;
        workRe[u] += tRe;
        workIm[u] += tIm;
      }
    }
  }

  const magnitude = new Float32Array(nFreqs);
  for (let k = 0; k < nFreqs; k++) {
    magnitude[k] = Math.sqrt(workRe[k] * workRe[k] + workIm[k] * workIm[k]);
  }
  return magnitude;
}

export function computeMelSpectrogram(audioData: Float32Array, sampleRate: number) {
  const {
    sample_rate: targetSr,
    n_fft: nFft,
    win_length: winLength,
    hop_length: hopLength,
    preemph,
    log_zero_guard: logZeroGuard,
    n_mels: nMels,
  } = DEFAULT_MEL_CONFIG;

  if (!melFilterbank) {
    melFilterbank = createMelFilterbank(targetSr, nFft, nMels, DEFAULT_MEL_CONFIG.fmin, DEFAULT_MEL_CONFIG.fmax);
  }

  const audio = resampleAudio(audioData, sampleRate, targetSr);
  const audioPreemph = new Float32Array(audio.length);
  audioPreemph[0] = audio[0];
  for (let i = 1; i < audio.length; i++) {
    audioPreemph[i] = audio[i] - preemph * audio[i - 1];
  }

  const padAmount = Math.floor(nFft / 2);
  const audioPadded = new Float32Array(audio.length + 2 * padAmount);
  audioPadded.set(audioPreemph, padAmount);

  const numFrames = 1 + Math.floor((audioPadded.length - nFft) / hopLength);
  const nFreqs = Math.floor(nFft / 2) + 1;

  const hannWindow = createHannWindow(winLength);
  const padLeft = Math.floor((nFft - winLength) / 2);
  const paddedWindow = new Float32Array(nFft);
  for (let i = 0; i < winLength; i++) {
    paddedWindow[padLeft + i] = hannWindow[i];
  }

  const melFeatures = new Float32Array(numFrames * nMels);

  for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
    const start = frameIdx * hopLength;
    const frame = new Float32Array(nFft);
    for (let i = 0; i < nFft; i++) {
      frame[i] = audioPadded[start + i] * paddedWindow[i];
    }
    const magnitude = computeRfftMagnitude(frame);
    for (let m = 0; m < nMels; m++) {
      let melVal = 0;
      for (let k = 0; k < nFreqs; k++) {
        melVal += melFilterbank[m][k] * magnitude[k] * magnitude[k];
      }
      melFeatures[frameIdx * nMels + m] = Math.log(Math.max(melVal, logZeroGuard));
    }
  }

  if (DEFAULT_MEL_CONFIG.normalize === 'per_feature') {
    for (let m = 0; m < nMels; m++) {
      let mean = 0;
      let std = 0;
      for (let t = 0; t < numFrames; t++) {
        mean += melFeatures[t * nMels + m];
      }
      mean /= numFrames;
      for (let t = 0; t < numFrames; t++) {
        const diff = melFeatures[t * nMels + m] - mean;
        std += diff * diff;
      }
      std = Math.sqrt(std / numFrames + 1e-5);
      for (let t = 0; t < numFrames; t++) {
        melFeatures[t * nMels + m] = (melFeatures[t * nMels + m] - mean) / std;
      }
    }
  }

  return { melFeatures, numFrames };
}

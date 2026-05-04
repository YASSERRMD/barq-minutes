import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mic, MicOff, Square, Upload } from 'lucide-react';
import ProgressTimeline, { type ProgressStep } from '../components/ProgressTimeline';
import Waveform from '../components/Waveform';
import { extractActionItems } from '../pipeline/extractActionItems';
import { extractDecisions } from '../pipeline/extractDecisions';
import { extractQuestions } from '../pipeline/extractQuestions';
import { chunkTranscriptForExtraction } from '../pipeline/chunker';
import { dedupeStructuredItems } from '../pipeline/dedupe';
import { generateFinalSummary } from '../pipeline/summarize';
import { ASR_SAMPLE_RATE, transcribeAudioBlob, transcribeAudioSamples } from '../pipeline/transcribe';
import { saveAudioBlob } from '../storage/audio';
import { saveMeeting } from '../storage/meetings';
import { indexMeetingForAsk } from '../pipeline/askMeeting';
import { formatClock } from '../utils/time';
import { estimateTokens } from '../utils/tokens';
import { ulid } from '../utils/ulid';
import { useModelBoot } from '../context/ModelBootContext';
import type { TranscriptTurn } from '../schemas/meeting';

type RecordingState = 'idle' | 'recording' | 'stopping' | 'processing';

const LIVE_TRANSCRIPTION_INTERVAL_MS = 10_000;
const MIN_LIVE_TRANSCRIPTION_SEC = 3;

function transcriptFromText(text: string, durationSec: number): TranscriptTurn[] {
  const cleanText = text.trim();
  if (!cleanText) return [];
  return [
    {
      speaker: 'Speaker 1',
      text: cleanText,
      startSec: 0,
      endSec: Math.max(1, durationSec),
    },
  ];
}

function transcriptText(turns: TranscriptTurn[]) {
  return turns.map((turn) => turn.text.trim()).filter(Boolean).join('\n');
}

function concatSamples(chunks: Float32Array[], totalSamples: number) {
  const samples = new Float32Array(totalSamples);
  let offset = 0;
  for (const chunk of chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  return samples;
}

export default function Recording() {
  const navigate = useNavigate();
  const {
    asrSession,
    ensureAsrSession,
    ensureLlmSession,
    ensureEmbeddingSession,
    state: modelState,
    allReady,
  } = useModelBoot();
  const recorderRef = useRef<MediaRecorder | null>(null);
  const activeAsrSessionRef = useRef<typeof asrSession>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const silenceGainRef = useRef<GainNode | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const pendingLiveSampleChunksRef = useRef<Float32Array[]>([]);
  const pendingLiveSampleCountRef = useRef(0);
  const allLiveSampleChunksRef = useRef<Float32Array[]>([]);
  const allLiveSampleCountRef = useRef(0);
  const liveSampleRateRef = useRef(ASR_SAMPLE_RATE);
  const partialTurnsRef = useRef<TranscriptTurn[]>([]);
  const startedAtRef = useRef<number>(0);
  const liveTimerRef = useRef<number | null>(null);
  const liveTranscribingRef = useRef(false);
  const liveTranscriptionPromiseRef = useRef<Promise<void> | null>(null);
  const pendingLiveStartSecRef = useRef(0);
  const processedLiveSecRef = useRef(0);
  const partialTextRef = useRef('');

  const [state, setState] = useState<RecordingState>('idle');
  const [title, setTitle] = useState('Untitled meeting');
  const [storeAudio, setStoreAudio] = useState(false);
  const [muted, setMuted] = useState(false);
  const [durationSec, setDurationSec] = useState(0);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [partialText, setPartialText] = useState('');
  const [status, setStatus] = useState('Ready to record');
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([
    { id: 'transcribe', label: 'Transcribing audio', detail: 'Waiting for recording', status: 'pending', progress: 0 },
    { id: 'extract', label: 'Extracting structured items', detail: 'Waiting for chunks', status: 'pending' },
    { id: 'dedupe', label: 'Deduplicating', detail: 'Waiting for extracted items', status: 'pending' },
    { id: 'summary', label: 'Generating summary', detail: 'Waiting for dedupe', status: 'pending' },
  ]);

  useEffect(() => {
    if (state !== 'recording') return;
    const timer = window.setInterval(() => {
      setDurationSec(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }, 500);
    return () => window.clearInterval(timer);
  }, [state]);

  useEffect(() => {
    partialTextRef.current = partialText;
  }, [partialText]);

  useEffect(() => {
    return () => {
      if (liveTimerRef.current !== null) {
        window.clearInterval(liveTimerRef.current);
      }
      processorRef.current?.disconnect();
      silenceGainRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      void audioContextRef.current?.close();
    };
  }, []);

  const tokenEstimate = useMemo(() => estimateTokens(partialText), [partialText]);
  const asrModelStatus = allReady
    ? 'All local models ready'
    : 'Preparing local model cache before recording or upload';
  const modelRows = [
    { key: 'ASR', state: modelState.asr },
    { key: 'Extraction', state: modelState.llm },
    { key: 'Search', state: modelState.embeddings },
  ];
  const canStartInput = state === 'idle' && allReady;

  async function runLiveTranscription(finalPass = false) {
    const session = activeAsrSessionRef.current ?? asrSession;
    if (!session) return;
    if (liveTranscribingRef.current) {
      if (finalPass && liveTranscriptionPromiseRef.current) {
        await liveTranscriptionPromiseRef.current;
      }
      return;
    }

    const pendingChunks = pendingLiveSampleChunksRef.current;
    const pendingSampleCount = pendingLiveSampleCountRef.current;
    const sampleRate = liveSampleRateRef.current;
    const pendingDurationSec = pendingSampleCount / sampleRate;
    if (pendingSampleCount === 0) return;
    if (!finalPass && pendingDurationSec < MIN_LIVE_TRANSCRIPTION_SEC) return;

    liveTranscribingRef.current = true;
    pendingLiveSampleChunksRef.current = [];
    pendingLiveSampleCountRef.current = 0;
    const segmentStartSec = pendingLiveStartSecRef.current;
    const samples = concatSamples(pendingChunks, pendingSampleCount);
    setStatus(finalPass ? 'Completing final live transcript segment' : 'Live transcription segment running');
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'transcribe'
        ? {
          ...step,
          status: 'active',
          detail: finalPass ? 'Completing remaining audio segment' : `${partialTurnsRef.current.length + 1} live segments`,
          progress: null,
        }
        : step
    )));

    const promise = transcribeAudioSamples(samples, sampleRate, session, {
      meetingTitle: title,
      allowEmpty: true,
      startSec: segmentStartSec,
      onProgress: finalPass ? (event) => setStatus(event.message) : undefined,
    })
      .then((turns) => {
        const shiftedTurns = turns.filter((turn) => turn.text.trim());

        if (shiftedTurns.length > 0) {
          partialTurnsRef.current = [...partialTurnsRef.current, ...shiftedTurns];
          const nextText = transcriptText(partialTurnsRef.current);
          partialTextRef.current = nextText;
          setPartialText(nextText);
        }

        processedLiveSecRef.current = Math.max(processedLiveSecRef.current, segmentStartSec + pendingDurationSec);
        pendingLiveStartSecRef.current = processedLiveSecRef.current;
        setStatus(finalPass ? 'Transcript complete' : 'Recording audio locally. Live transcript updated');
      })
      .catch((error) => {
        pendingLiveSampleChunksRef.current = [...pendingChunks, ...pendingLiveSampleChunksRef.current];
        pendingLiveSampleCountRef.current += pendingSampleCount;
        pendingLiveStartSecRef.current = segmentStartSec;
        console.warn('[Recording] Live transcription failed', error);
        setStatus(finalPass ? 'Final transcription failed' : 'Recording audio locally. Live transcript will retry');
      })
      .finally(() => {
        liveTranscribingRef.current = false;
        liveTranscriptionPromiseRef.current = null;
      });

    liveTranscriptionPromiseRef.current = promise;
    await promise;
  }

  async function startRecording() {
    if (!canStartInput) {
      setStatus('Local models are still preparing');
      return;
    }
    setStatus(asrSession ? 'ASR model ready' : 'Loading ASR model from browser cache');
    const session = asrSession ?? await ensureAsrSession();
    activeAsrSessionRef.current = session;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext({ sampleRate: ASR_SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(stream);
    const nextAnalyser = audioContext.createAnalyser();
    nextAnalyser.fftSize = 2048;
    source.connect(nextAnalyser);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const silenceGain = audioContext.createGain();
    silenceGain.gain.value = 0;
    processor.onaudioprocess = (event) => {
      if (recorderRef.current?.state !== 'recording') return;
      const input = event.inputBuffer.getChannelData(0);
      const samples = input.slice();
      pendingLiveSampleChunksRef.current.push(samples);
      pendingLiveSampleCountRef.current += samples.length;
      allLiveSampleChunksRef.current.push(samples);
      allLiveSampleCountRef.current += samples.length;
    };
    source.connect(processor);
    processor.connect(silenceGain);
    silenceGain.connect(audioContext.destination);

    chunksRef.current = [];
    pendingLiveSampleChunksRef.current = [];
    pendingLiveSampleCountRef.current = 0;
    allLiveSampleChunksRef.current = [];
    allLiveSampleCountRef.current = 0;
    partialTurnsRef.current = [];
    const recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };
    recorder.start(1000);

    recorderRef.current = recorder;
    streamRef.current = stream;
    audioContextRef.current = audioContext;
    processorRef.current = processor;
    silenceGainRef.current = silenceGain;
    liveSampleRateRef.current = audioContext.sampleRate;
    startedAtRef.current = Date.now();
    setDurationSec(0);
    setAnalyser(nextAnalyser);
    setPartialText('');
    partialTextRef.current = '';
    pendingLiveStartSecRef.current = 0;
    processedLiveSecRef.current = 0;
    setState('recording');
    setStatus('Recording audio locally. Live transcription will update every few seconds');
    setProgressSteps([
      { id: 'transcribe', label: 'Live transcription', detail: 'Waiting for first audio segment', status: 'active', progress: null },
      { id: 'extract', label: 'Extracting structured items', detail: 'Waiting for transcript', status: 'pending' },
      { id: 'dedupe', label: 'Deduplicating', detail: 'Waiting for extracted items', status: 'pending' },
      { id: 'summary', label: 'Generating summary', detail: 'Waiting for dedupe', status: 'pending' },
    ]);
    liveTimerRef.current = window.setInterval(() => {
      void runLiveTranscription(false);
    }, LIVE_TRANSCRIPTION_INTERVAL_MS);
  }

  async function processMeetingTranscript(params: {
    audioBlob: Blob;
    transcript: TranscriptTurn[];
    startedAt: number;
    endedAt: number;
    durationSec: number;
    transcriptDetail: string;
    meetingTitle?: string;
  }) {
    const id = ulid();
    const transcript = params.transcript.filter((turn) => turn.text.trim());

    setStatus('Processing transcript');
    setProgressSteps([
      { id: 'transcribe', label: 'Transcript ready', detail: params.transcriptDetail, status: 'done', progress: 100 },
      { id: 'extract', label: 'Extracting structured items', detail: 'Waiting for chunks', status: 'pending' },
      { id: 'dedupe', label: 'Deduplicating', detail: 'Waiting for extracted items', status: 'pending' },
      { id: 'summary', label: 'Generating summary', detail: 'Waiting for dedupe', status: 'pending' },
    ]);

    if (transcript.length === 0) {
      setStatus('No transcript text was produced');
      setProgressSteps((steps) => steps.map((step) => (
        step.id === 'transcribe' ? { ...step, status: 'done', detail: 'No transcript text found', progress: 100 } : step
      )));
      return;
    }

    const finalTranscriptText = transcriptText(transcript);
    partialTextRef.current = finalTranscriptText;
    partialTurnsRef.current = transcript;
    setPartialText(finalTranscriptText);

    const windows = chunkTranscriptForExtraction(transcript);
    const totalChunks = windows.length;
    const decisions = [];
    const actionItems = [];
    const openQuestions = [];

    setStatus('Loading extraction model from browser cache');
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'extract'
        ? { ...step, status: 'active', detail: modelState.llm.status === 'ready' ? 'Extraction model ready' : 'Loading extraction model', progress: null }
        : step
    )));
    await ensureLlmSession();

    for (const window of windows) {
      setStatus(`Extracting structured items ${window.index + 1}/${totalChunks}`);
      setProgressSteps((steps) => steps.map((step) => (
        step.id === 'extract'
          ? { ...step, status: 'active', detail: `${window.index + 1}/${totalChunks} transcript chunks`, progress: ((window.index + 1) / Math.max(1, totalChunks)) * 100 }
          : step
      )));
      const [d, a, q] = await Promise.all([
        extractDecisions(window, totalChunks),
        extractActionItems(window, totalChunks),
        extractQuestions(window, totalChunks),
      ]);
      decisions.push(...d);
      actionItems.push(...a);
      openQuestions.push(...q);
    }
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'extract' ? { ...step, status: 'done', detail: `${totalChunks}/${totalChunks} transcript chunks`, progress: 100 } : step
    )));

    setStatus('Deduplicating structured items');
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'dedupe' ? { ...step, status: 'active', detail: 'Loading embedding model', progress: null } : step
    )));
    await ensureEmbeddingSession();
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'dedupe' ? { ...step, status: 'active', detail: 'Clustering duplicate items', progress: null } : step
    )));
    const deduped = await dedupeStructuredItems({ decisions, actionItems, openQuestions });
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'dedupe' ? { ...step, status: 'done', detail: 'Done', progress: 100 } : step
    )));

    setStatus('Generating summary');
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'summary' ? { ...step, status: 'active', detail: 'Building complete meeting summary from transcript chunks', progress: null } : step
    )));
    const summary = await generateFinalSummary({
      transcript,
      ...deduped,
      onProgress: (event) => {
        setStatus(event.message);
        setProgressSteps((steps) => steps.map((step) => (
          step.id === 'summary'
            ? {
                ...step,
                status: 'active',
                detail: event.message,
                progress: event.totalChunks > 0
                  ? Math.min(95, ((event.chunkIndex + 1) / event.totalChunks) * 100)
                  : null,
              }
            : step
        )));
      },
    });
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'summary' ? { ...step, status: 'done', detail: 'Done', progress: 100 } : step
    )));

    const audioBlobKey = storeAudio ? await saveAudioBlob(id, params.audioBlob) : null;
    const meeting = await saveMeeting({
      id,
      title: params.meetingTitle?.trim() || title.trim() || 'Untitled meeting',
      startedAt: params.startedAt,
      endedAt: params.endedAt,
      durationSec: params.durationSec,
      participants: [],
      audioBlobKey,
      transcript,
      ...deduped,
      summary,
      tags: [],
    });

    setStatus('Indexing transcript for search');
    await indexMeetingForAsk(meeting);
    navigate(`/meeting/${id}`);
  }

  async function stopRecording() {
    const recorder = recorderRef.current;
    if (!recorder || state !== 'recording') return;

    setState('stopping');
    setStatus('Recording stopped. Finalizing transcript');
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'transcribe'
        ? { ...step, status: 'active', detail: 'Stopping recorder and finalizing live transcript', progress: 10 }
        : step
    )));

    if (liveTimerRef.current !== null) {
      window.clearInterval(liveTimerRef.current);
      liveTimerRef.current = null;
    }

    const stopped = new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
    });
    recorder.stop();
    await stopped;

    processorRef.current?.disconnect();
    silenceGainRef.current?.disconnect();
    processorRef.current = null;
    silenceGainRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    await audioContextRef.current?.close();
    audioContextRef.current = null;
    streamRef.current = null;
    setAnalyser(null);
    setMuted(false);
    setState('processing');

    if (liveTranscriptionPromiseRef.current) {
      await liveTranscriptionPromiseRef.current;
    }

    const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
    const endedAt = Date.now();
    const duration = Math.max(1, Math.floor((endedAt - startedAtRef.current) / 1000));

    await runLiveTranscription(true);

    const liveText = transcriptText(partialTurnsRef.current);
    const editedText = partialTextRef.current.trim();
    let transcript = editedText && editedText !== liveText
      ? transcriptFromText(editedText, duration)
      : partialTurnsRef.current;
    const transcriptEndSec = Math.max(0, ...transcript.map((turn) => turn.endSec));
    let transcriptDetail = `${transcript.length} live transcript segments`;

    if (!editedText && duration >= 30 && transcriptEndSec < duration * 0.6) {
      const session = activeAsrSessionRef.current ?? asrSession ?? await ensureAsrSession();
      const fullSamples = concatSamples(allLiveSampleChunksRef.current, allLiveSampleCountRef.current);
      setStatus('Live transcript coverage is incomplete. Recovering full transcript');
      setProgressSteps((steps) => steps.map((step) => (
        step.id === 'transcribe'
          ? { ...step, status: 'active', detail: 'Recovering full transcript', progress: 5 }
          : step
      )));
      transcript = await transcribeAudioSamples(fullSamples, liveSampleRateRef.current, session, {
        meetingTitle: title,
        fallbackText: liveText,
        onProgress: (event) => {
          setStatus(event.message);
          setProgressSteps((steps) => steps.map((step) => (
            step.id === 'transcribe'
              ? { ...step, status: 'active', detail: event.message, progress: event.progress }
              : step
          )));
        },
      });
      transcriptDetail = 'Recovered full transcript';
    }

    await processMeetingTranscript({
      audioBlob: blob,
      transcript,
      startedAt: startedAtRef.current,
      endedAt,
      durationSec: duration,
      transcriptDetail,
    });
  }

  async function importAudioFile(file: File | null) {
    if (!file || !canStartInput) return;

    setState('processing');
    setStatus(asrSession ? `Transcribing ${file.name}` : 'Loading ASR model from browser cache');
    const session = asrSession ?? await ensureAsrSession();
    activeAsrSessionRef.current = session;
    const nextTitle = title.trim() && title !== 'Untitled meeting'
      ? title
      : file.name.replace(/\.[^.]+$/, '') || 'Imported meeting';
    setTitle(nextTitle);
    setStatus(`Transcribing ${file.name}`);
    setProgressSteps([
      { id: 'transcribe', label: 'Transcribing audio', detail: 'Decoding imported audio file', status: 'active', progress: 5 },
      { id: 'extract', label: 'Extracting structured items', detail: 'Waiting for transcript', status: 'pending' },
      { id: 'dedupe', label: 'Deduplicating', detail: 'Waiting for extracted items', status: 'pending' },
      { id: 'summary', label: 'Generating summary', detail: 'Waiting for dedupe', status: 'pending' },
    ]);

    const startedAt = Date.now();
    const transcript = await transcribeAudioBlob(file, session, {
      meetingTitle: nextTitle,
      onProgress: (event) => {
        setStatus(event.message);
        setProgressSteps((steps) => steps.map((step) => (
          step.id === 'transcribe'
            ? { ...step, status: 'active', detail: event.message, progress: event.progress }
            : step
        )));
      },
    });
    const duration = Math.max(1, Math.ceil(Math.max(...transcript.map((turn) => turn.endSec), 1)));
    const endedAt = startedAt + duration * 1000;

    await processMeetingTranscript({
      audioBlob: file,
      transcript,
      startedAt,
      endedAt,
      durationSec: duration,
      transcriptDetail: `Imported ${file.type || 'audio file'}`,
      meetingTitle: nextTitle,
    });
  }

  function toggleMute() {
    const nextMuted = !muted;
    streamRef.current?.getAudioTracks().forEach((track) => {
      track.enabled = !nextMuted;
    });
    setMuted(nextMuted);
  }

  return (
    <section>
      <header className="page-header">
        <div>
          <p className="page-kicker">Recorder</p>
          <h1 className="page-title">New meeting</h1>
          <p className="page-subtitle">Audio stays in the browser. Audio blob storage is opt-in.</p>
        </div>
        <div className="record-stats">
          <span>{formatClock(durationSec)}</span>
          <span>{tokenEstimate} estimated tokens</span>
        </div>
      </header>

      <div className="record-layout">
        <section className="panel record-main">
          <label>
            <span className="field-label">Meeting title</span>
            <input value={title} onChange={(event) => setTitle(event.target.value)} disabled={state !== 'idle'} />
          </label>
          <p className="model-inline-status">{asrModelStatus}</p>

          <div className="model-prep-box">
            <h2 className="section-title">Local model cache</h2>
            <div className="model-prep-list">
              {modelRows.map((row) => (
                <div key={row.key} className="model-prep-row">
                  <span>{row.key}</span>
                  <strong>{row.state.status === 'ready' ? 'Ready' : row.state.status === 'error' ? 'Error' : row.state.message}</strong>
                </div>
              ))}
            </div>
          </div>

          <Waveform analyser={analyser} />

          <div className="record-controls">
            {state === 'idle' ? (
              <button className="button primary" type="button" onClick={startRecording} disabled={!canStartInput}>
                <Mic size={18} />
                {allReady ? 'Record' : 'Preparing models'}
              </button>
            ) : state === 'recording' ? (
              <>
                <button className="icon-button" type="button" onClick={toggleMute} aria-label={muted ? 'Unmute mic' : 'Mute mic'}>
                  {muted ? <MicOff size={18} /> : <Mic size={18} />}
                </button>
                <button className="button danger" type="button" onClick={stopRecording}>
                  <Square size={16} />
                  Stop
                </button>
              </>
            ) : (
              <button className="button danger" type="button" disabled>
                <Square size={16} />
                {state === 'stopping' ? 'Stopping' : 'Processing'}
              </button>
            )}
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={storeAudio}
                onChange={(event) => setStoreAudio(event.target.checked)}
                disabled={state !== 'idle'}
              />
              Store audio blob
            </label>
          </div>

          <div className="upload-box">
            <div>
              <h2 className="section-title">Import audio</h2>
              <p>Upload MP3, OGG, WAV, M4A, WebM, or FLAC and process it with the same local pipeline.</p>
            </div>
            <label className={`button ${canStartInput ? '' : 'disabled'}`}>
              <Upload size={18} />
              {allReady ? 'Choose audio' : 'Preparing models'}
              <input
                type="file"
                accept="audio/*,.mp3,.ogg,.oga,.wav,.m4a,.webm,.flac"
                disabled={!canStartInput}
                onChange={(event) => {
                  void importAudioFile(event.target.files?.[0] ?? null);
                  event.target.value = '';
                }}
              />
            </label>
          </div>
        </section>

        <aside className="panel">
          <h2 className="section-title">Live transcription</h2>
          <textarea
            value={partialText}
            onChange={(event) => setPartialText(event.target.value)}
            placeholder="Live transcript appears here while recording. You can type corrections before stopping."
            disabled={state !== 'recording'}
          />
          <p className="status-line">{status}</p>
          <ProgressTimeline steps={progressSteps} />
        </aside>
      </div>
    </section>
  );
}

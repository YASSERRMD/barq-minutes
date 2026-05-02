import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mic, MicOff, Square } from 'lucide-react';
import ProgressTimeline, { type ProgressStep } from '../components/ProgressTimeline';
import Waveform from '../components/Waveform';
import { extractActionItems } from '../pipeline/extractActionItems';
import { extractDecisions } from '../pipeline/extractDecisions';
import { extractQuestions } from '../pipeline/extractQuestions';
import { chunkTranscriptForExtraction } from '../pipeline/chunker';
import { dedupeStructuredItems } from '../pipeline/dedupe';
import { generateFinalSummary } from '../pipeline/summarize';
import { transcribeAudioBlob } from '../pipeline/transcribe';
import { saveAudioBlob } from '../storage/audio';
import { saveMeeting } from '../storage/meetings';
import { indexMeetingForAsk } from '../pipeline/askMeeting';
import { formatClock } from '../utils/time';
import { estimateTokens } from '../utils/tokens';
import { ulid } from '../utils/ulid';
import { useModelBoot } from '../context/ModelBootContext';

type RecordingState = 'idle' | 'recording' | 'processing';

export default function Recording() {
  const navigate = useNavigate();
  const { asrSession } = useModelBoot();
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const startedAtRef = useRef<number>(0);

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

  const tokenEstimate = useMemo(() => estimateTokens(partialText), [partialText]);

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const nextAnalyser = audioContext.createAnalyser();
    nextAnalyser.fftSize = 2048;
    source.connect(nextAnalyser);

    chunksRef.current = [];
    const recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.start(1000);

    recorderRef.current = recorder;
    streamRef.current = stream;
    audioContextRef.current = audioContext;
    startedAtRef.current = Date.now();
    setDurationSec(0);
    setAnalyser(nextAnalyser);
    setState('recording');
    setStatus('Recording audio locally');
  }

  async function stopRecording() {
    const recorder = recorderRef.current;
    if (!recorder) return;

    setState('processing');
    setStatus('Stopping recorder');

    const stopped = new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
    });
    recorder.stop();
    await stopped;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    await audioContextRef.current?.close();

    const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
    const id = ulid();
    const endedAt = Date.now();
    const duration = Math.max(1, Math.floor((endedAt - startedAtRef.current) / 1000));

    setStatus('Transcribing audio locally');
    setProgressSteps([
      { id: 'transcribe', label: 'Transcribing audio', detail: 'Decoding and running ASR inference', status: 'active', progress: 5 },
      { id: 'extract', label: 'Extracting structured items', detail: 'Waiting for chunks', status: 'pending' },
      { id: 'dedupe', label: 'Deduplicating', detail: 'Waiting for extracted items', status: 'pending' },
      { id: 'summary', label: 'Generating summary', detail: 'Waiting for dedupe', status: 'pending' },
    ]);
    if (!asrSession) {
      console.error('[Recording] asrSession is null. Boot gate should have prevented this');
      return;
    }
    const transcript = await transcribeAudioBlob(blob, asrSession, {
      meetingTitle: title,
      fallbackText: partialText,
      onProgress: (event) => setStatus(event.message),
    });
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'transcribe' ? { ...step, status: 'done', detail: 'Transcript ready', progress: 100 } : step
    )));

    const windows = chunkTranscriptForExtraction(transcript);
    const totalChunks = windows.length;
    const decisions = [];
    const actionItems = [];
    const openQuestions = [];

    for (const window of windows) {
      setStatus(`Extracting structured items ${window.index + 1}/${totalChunks}`);
      setProgressSteps((steps) => steps.map((step) => (
        step.id === 'extract'
          ? { ...step, status: 'active', detail: `${window.index + 1}/${totalChunks} chunks`, progress: ((window.index + 1) / Math.max(1, totalChunks)) * 100 }
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
      step.id === 'extract' ? { ...step, status: 'done', detail: `${totalChunks}/${totalChunks} chunks`, progress: 100 } : step
    )));

    setStatus('Deduplicating structured items');
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'dedupe' ? { ...step, status: 'active', detail: 'Clustering duplicate items', progress: null } : step
    )));
    const deduped = await dedupeStructuredItems({ decisions, actionItems, openQuestions });
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'dedupe' ? { ...step, status: 'done', detail: 'Done', progress: 100 } : step
    )));

    setStatus('Generating summary');
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'summary' ? { ...step, status: 'active', detail: 'Creating 5 executive bullets', progress: null } : step
    )));
    const summary = await generateFinalSummary({ transcript, ...deduped });
    setProgressSteps((steps) => steps.map((step) => (
      step.id === 'summary' ? { ...step, status: 'done', detail: 'Done', progress: 100 } : step
    )));

    const audioBlobKey = storeAudio ? await saveAudioBlob(id, blob) : null;
    const meeting = await saveMeeting({
      id,
      title: title.trim() || 'Untitled meeting',
      startedAt: startedAtRef.current,
      endedAt,
      durationSec: duration,
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
            <input value={title} onChange={(event) => setTitle(event.target.value)} disabled={state === 'processing'} />
          </label>

          <Waveform analyser={analyser} />

          <div className="record-controls">
            {state === 'idle' ? (
              <button className="button primary" type="button" onClick={startRecording}>
                <Mic size={18} />
                Record
              </button>
            ) : (
              <>
                <button className="icon-button" type="button" onClick={toggleMute} aria-label={muted ? 'Unmute mic' : 'Mute mic'}>
                  {muted ? <MicOff size={18} /> : <Mic size={18} />}
                </button>
                <button className="button danger" type="button" onClick={stopRecording} disabled={state === 'processing'}>
                  <Square size={16} />
                  Stop
                </button>
              </>
            )}
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={storeAudio}
                onChange={(event) => setStoreAudio(event.target.checked)}
                disabled={state === 'processing'}
              />
              Store audio blob
            </label>
          </div>
        </section>

        <aside className="panel">
          <h2 className="section-title">Live transcription</h2>
          <textarea
            value={partialText}
            onChange={(event) => setPartialText(event.target.value)}
            placeholder="Partial transcript appears here. You can type corrections while recording."
            disabled={state === 'processing'}
          />
          <p className="status-line">{status}</p>
          <ProgressTimeline steps={progressSteps} />
        </aside>
      </div>
    </section>
  );
}

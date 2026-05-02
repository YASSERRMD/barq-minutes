import type { TranscriptTurn } from '../schemas/meeting';

const COLLECTION_PREFIX = 'barq-minutes';

export type TranscriptChunk = {
  id: string;
  meetingId: string;
  text: string;
  startSec: number;
  endSec: number;
  turnIndexes: number[];
};

export type RetrievedChunk = TranscriptChunk & {
  score: number;
};

type BarqVWebCtor = new (collectionName: string, modelUrl?: string | null) => {
  backend_info(): string;
  clear(): Promise<unknown>;
  count(): number;
  insert_texts(texts: string[], metadata: TranscriptChunk[]): Promise<unknown>;
  load(): Promise<unknown>;
  save(): Promise<unknown>;
  search(query: string, topK: number, hybrid: boolean): Promise<unknown>;
};

let initPromise: Promise<void> | null = null;
const stores = new Map<string, InstanceType<BarqVWebCtor>>();

async function initBarqRuntime() {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const [vweb, wasm] = await Promise.all([import('barq-vweb'), import('barq-wasm')]);
    await (vweb as any).default();
    await (wasm as any).default();
  })();

  return initPromise;
}

function collectionName(meetingId: string) {
  return `${COLLECTION_PREFIX}:${meetingId}`;
}

export async function getVectorStore(meetingId: string) {
  await initBarqRuntime();
  const name = collectionName(meetingId);
  const existing = stores.get(name);
  if (existing) return existing;

  const module = await import('barq-vweb');
  const Store = (module as any).BarqVWeb as BarqVWebCtor;
  const store = new Store(name);
  await store.load();
  stores.set(name, store);
  return store;
}

export function createTranscriptChunks(
  meetingId: string,
  transcript: TranscriptTurn[],
  charsPerChunk = 2048,
  overlapChars = 256,
): TranscriptChunk[] {
  const chunks: TranscriptChunk[] = [];
  let buffer = '';
  let startSec = 0;
  let turnIndexes: number[] = [];

  for (let index = 0; index < transcript.length; index++) {
    const turn = transcript[index];
    if (!buffer) startSec = turn.startSec;
    const line = `[${Math.round(turn.startSec)}s] ${turn.speaker}: ${turn.text}`;

    if (buffer.length + line.length > charsPerChunk && buffer) {
      const lastTurn = transcript[turnIndexes[turnIndexes.length - 1]];
      chunks.push({
        id: `${meetingId}:chunk:${chunks.length}`,
        meetingId,
        text: buffer.trim(),
        startSec,
        endSec: lastTurn?.endSec ?? startSec,
        turnIndexes,
      });
      buffer = buffer.slice(Math.max(0, buffer.length - overlapChars));
      turnIndexes = [];
      startSec = turn.startSec;
    }

    buffer += `${line}\n`;
    turnIndexes.push(index);
  }

  if (buffer.trim()) {
    const lastTurn = transcript[turnIndexes[turnIndexes.length - 1]];
    chunks.push({
      id: `${meetingId}:chunk:${chunks.length}`,
      meetingId,
      text: buffer.trim(),
      startSec,
      endSec: lastTurn?.endSec ?? startSec,
      turnIndexes,
    });
  }

  return chunks;
}

export async function indexTranscriptChunks(chunks: TranscriptChunk[]): Promise<number> {
  if (chunks.length === 0) return 0;
  const meetingId = chunks[0].meetingId;
  const store = await getVectorStore(meetingId);
  await store.clear();
  await store.insert_texts(
    chunks.map((chunk) => chunk.text),
    chunks,
  );
  await store.save();
  return store.count();
}

export async function searchMeetingChunks(
  meetingId: string,
  query: string,
  topK = 5,
): Promise<RetrievedChunk[]> {
  const store = await getVectorStore(meetingId);
  const raw = await store.search(query, topK, true);
  const results = typeof raw === 'string' ? JSON.parse(raw) : raw;
  if (!Array.isArray(results)) return [];

  return results
    .map((result: any) => {
      const metadata = result.metadata as TranscriptChunk | undefined;
      if (!metadata) return null;
      return {
        ...metadata,
        score: Number(result.score ?? 0),
      };
    })
    .filter(Boolean) as RetrievedChunk[];
}

export async function clearMeetingVectors(meetingId: string): Promise<void> {
  const store = await getVectorStore(meetingId);
  await store.clear();
  await store.save();
}

export async function getVectorBackendInfo(meetingId: string): Promise<string> {
  const store = await getVectorStore(meetingId);
  return store.backend_info();
}

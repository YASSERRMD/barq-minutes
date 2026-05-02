import type { ActionItem, Decision, OpenQuestion } from '../schemas/meeting';
import { loadEmbeddingSession } from '../models/embedSession';

const SIMILARITY_THRESHOLD = 0.85;
const STOPWORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'that',
  'the',
  'to',
  'we',
  'will',
  'with',
]);

type StructuredItem = Decision | ActionItem | OpenQuestion;

function stem(token: string): string {
  return token
    .replace(/(ing|ed|es|s)$/i, '')
    .replace(/(ize|ise)$/i, '');
}

export function normalizeForDedupe(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((token) => token && !STOPWORDS.has(token))
    .map((token, index) => (index === 0 ? stem(token) : token))
    .join(' ');
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aNorm += a[i] * a[i];
    bNorm += b[i] * b[i];
  }
  if (aNorm === 0 || bNorm === 0) return 0;
  return dot / (Math.sqrt(aNorm) * Math.sqrt(bNorm));
}

function mergeDecision(cluster: Decision[]): Decision {
  const canonical = cluster.reduce((best, item) => (item.text.length > best.text.length ? item : best));
  return {
    ...canonical,
    rationale: canonical.rationale ?? cluster.find((item) => item.rationale)?.rationale ?? null,
    speaker: canonical.speaker ?? cluster.find((item) => item.speaker)?.speaker ?? null,
    timestampSec: Math.min(...cluster.map((item) => item.timestampSec)),
  };
}

function mergeActionItem(cluster: ActionItem[]): ActionItem {
  const canonical = cluster.reduce((best, item) => (item.text.length > best.text.length ? item : best));
  return {
    ...canonical,
    owner: canonical.owner ?? cluster.find((item) => item.owner)?.owner ?? null,
    dueDate: canonical.dueDate ?? cluster.find((item) => item.dueDate)?.dueDate ?? null,
    timestampSec: Math.min(...cluster.map((item) => item.timestampSec)),
  };
}

function mergeOpenQuestion(cluster: OpenQuestion[]): OpenQuestion {
  const canonical = cluster.reduce((best, item) => (item.text.length > best.text.length ? item : best));
  return {
    ...canonical,
    raisedBy: canonical.raisedBy ?? cluster.find((item) => item.raisedBy)?.raisedBy ?? null,
    timestampSec: Math.min(...cluster.map((item) => item.timestampSec)),
  };
}

async function clusterItems<T extends StructuredItem>(items: T[]): Promise<T[][]> {
  if (items.length <= 1) return items.map((item) => [item]);
  const embeddingSession = await loadEmbeddingSession();
  const normalized = items.map((item) => normalizeForDedupe(item.text));
  const embeddings = await embeddingSession.embed(normalized);
  const visited = new Set<number>();
  const clusters: T[][] = [];

  for (let i = 0; i < items.length; i++) {
    if (visited.has(i)) continue;
    visited.add(i);
    const cluster = [items[i]];

    for (let j = i + 1; j < items.length; j++) {
      if (visited.has(j)) continue;
      if (cosine(embeddings[i], embeddings[j]) >= SIMILARITY_THRESHOLD) {
        visited.add(j);
        cluster.push(items[j]);
      }
    }

    clusters.push(cluster);
  }

  return clusters;
}

export async function dedupeDecisions(items: Decision[]): Promise<Decision[]> {
  const clusters = await clusterItems(items);
  return clusters.map((cluster) => mergeDecision(cluster)).sort((a, b) => a.timestampSec - b.timestampSec);
}

export async function dedupeActionItems(items: ActionItem[]): Promise<ActionItem[]> {
  const clusters = await clusterItems(items);
  return clusters.map((cluster) => mergeActionItem(cluster)).sort((a, b) => a.timestampSec - b.timestampSec);
}

export async function dedupeOpenQuestions(items: OpenQuestion[]): Promise<OpenQuestion[]> {
  const clusters = await clusterItems(items);
  return clusters.map((cluster) => mergeOpenQuestion(cluster)).sort((a, b) => a.timestampSec - b.timestampSec);
}

export async function dedupeStructuredItems(input: {
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}) {
  const [decisions, actionItems, openQuestions] = await Promise.all([
    dedupeDecisions(input.decisions),
    dedupeActionItems(input.actionItems),
    dedupeOpenQuestions(input.openQuestions),
  ]);

  return { decisions, actionItems, openQuestions };
}

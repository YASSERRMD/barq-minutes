import type { TranscriptTurn } from '../schemas/meeting';
import { estimateTokens } from '../utils/tokens';

export type TranscriptWindow = {
  index: number;
  text: string;
  startSec: number;
  endSec: number;
  turnIndexes: number[];
  estimatedTokens: number;
};

export const EXTRACTION_WINDOW_CHARS = 6000;
export const EXTRACTION_OVERLAP_CHARS = 600;
export const BOUNDARY_SEARCH_CHARS = 200;

function turnToLine(turn: TranscriptTurn, index: number) {
  return `[turn ${index} | ${Math.round(turn.startSec)}s-${Math.round(turn.endSec)}s] ${turn.speaker}: ${turn.text}`;
}

function findBoundary(text: string, target: number): number {
  const min = Math.max(0, target - BOUNDARY_SEARCH_CHARS);
  const max = Math.min(text.length, target + BOUNDARY_SEARCH_CHARS);
  const slice = text.slice(min, max);

  const speakerMatches = [...slice.matchAll(/\n\[turn\s+\d+\s+\|/g)];
  if (speakerMatches.length > 0) {
    const nearest = speakerMatches.reduce((best, match) => {
      const absolute = min + (match.index ?? 0) + 1;
      return Math.abs(absolute - target) < Math.abs(best - target) ? absolute : best;
    }, min + (speakerMatches[0].index ?? 0) + 1);
    return nearest;
  }

  const sentenceMatches = [...slice.matchAll(/[.!?]\s+/g)];
  if (sentenceMatches.length > 0) {
    const nearest = sentenceMatches.reduce((best, match) => {
      const absolute = min + (match.index ?? 0) + match[0].length;
      return Math.abs(absolute - target) < Math.abs(best - target) ? absolute : best;
    }, min + (sentenceMatches[0].index ?? 0) + sentenceMatches[0].length);
    return nearest;
  }

  return target;
}

function parseTurnIndexes(text: string): number[] {
  return [...text.matchAll(/\[turn\s+(\d+)\s+\|/g)].map((match) => Number(match[1]));
}

export function serializeTranscript(transcript: TranscriptTurn[]): string {
  return transcript.map((turn, index) => turnToLine(turn, index)).join('\n');
}

export function chunkTranscriptForExtraction(
  transcript: TranscriptTurn[],
  windowChars = EXTRACTION_WINDOW_CHARS,
  overlapChars = EXTRACTION_OVERLAP_CHARS,
): TranscriptWindow[] {
  const text = serializeTranscript(transcript);
  if (!text.trim()) return [];

  const windows: TranscriptWindow[] = [];
  let start = 0;

  while (start < text.length) {
    const targetEnd = Math.min(text.length, start + windowChars);
    const end = targetEnd >= text.length ? text.length : findBoundary(text, targetEnd);
    const windowText = text.slice(start, end).trim();
    const turnIndexes = parseTurnIndexes(windowText);
    const firstTurn = transcript[turnIndexes[0]];
    const lastTurn = transcript[turnIndexes[turnIndexes.length - 1]];

    windows.push({
      index: windows.length,
      text: windowText,
      startSec: firstTurn?.startSec ?? 0,
      endSec: lastTurn?.endSec ?? firstTurn?.endSec ?? 0,
      turnIndexes,
      estimatedTokens: estimateTokens(windowText),
    });

    if (end >= text.length) break;
    start = Math.max(0, end - overlapChars);
  }

  return windows;
}

export function chunkTranscriptForRag(
  transcript: TranscriptTurn[],
  chunkTokens = 512,
  overlapTokens = 64,
): TranscriptWindow[] {
  return chunkTranscriptForExtraction(transcript, chunkTokens * 4, overlapTokens * 4);
}

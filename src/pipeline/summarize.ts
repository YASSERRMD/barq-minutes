import { z } from 'zod';
import { loadLlmSession } from '../models/llmSession';
import type { ActionItem, Decision, OpenQuestion, TranscriptTurn } from '../schemas/meeting';
import { estimateTokens } from '../utils/tokens';
import { chunkTranscriptForExtraction, type TranscriptWindow } from './chunker';

const SUMMARY_CONTEXT_TOKEN_LIMIT = 2800;
const CHUNK_SUMMARY_MAX_NEW_TOKENS = 384;
const COMPLETE_SUMMARY_MAX_NEW_TOKENS = 768;
const CompleteSummarySchema = z.array(z.string().min(1)).min(3).max(12);
const ChunkSummarySchema = z.array(z.string().min(1)).min(3).max(8);

export type SummaryProgress = (event: {
  step: 'chunk_summary' | 'merge_summary' | 'final_summary';
  chunkIndex: number;
  totalChunks: number;
  message: string;
}) => void;

function extractJsonArray(text: string): unknown {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced?.[1] ?? text;
  const start = candidate.indexOf('[');
  const end = candidate.lastIndexOf(']');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('Summary response did not include a JSON array');
  }
  return JSON.parse(candidate.slice(start, end + 1));
}

function trimText(text: string, maxChars: number) {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= maxChars) return clean;
  return `${clean.slice(0, Math.max(0, maxChars - 3)).trim()}...`;
}

export function buildFallbackSummary(input: {
  transcript: TranscriptTurn[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}) {
  const windows = chunkTranscriptForExtraction(input.transcript, 3200, 0);
  const duration = input.transcript.length > 0
    ? Math.round(Math.max(...input.transcript.map((turn) => turn.endSec)))
    : 0;

  const discussionPoints = windows
    .slice(0, 10)
    .map((window) => {
      const text = window.text.replace(/\[turn\s+\d+\s+\|\s+\d+s-\d+s\]\s+/g, '');
      return `${Math.round(window.startSec)}s to ${Math.round(window.endSec)}s: ${trimText(text, 260)}`;
    })
    .filter(Boolean);

  if (discussionPoints.length === 0) {
    return ['The meeting transcript was captured but did not contain enough text for a detailed summary.'];
  }

  return [
    duration > 0 ? `The meeting covered approximately ${Math.ceil(duration / 60)} minutes of discussion.` : 'The meeting transcript was processed locally.',
    ...discussionPoints,
  ].slice(0, 12);
}

function summaryLine(window: TranscriptWindow, bullets: string[]) {
  return [
    `Chunk ${window.index + 1} (${Math.round(window.startSec)}s-${Math.round(window.endSec)}s):`,
    ...bullets.map((bullet) => `- ${bullet}`),
  ].join('\n');
}

function buildSummaryContext(input: {
  chunkSummaries: string[];
}) {
  const context = `Transcript chunk summaries:\n${input.chunkSummaries.join('\n\n')}`;

  if (estimateTokens(context) <= SUMMARY_CONTEXT_TOKEN_LIMIT) return context;

  const maxChars = SUMMARY_CONTEXT_TOKEN_LIMIT * 4;
  return trimText(context, maxChars);
}

async function summarizeWindow(
  window: TranscriptWindow,
  totalChunks: number,
  onProgress?: SummaryProgress,
) {
  const llm = await loadLlmSession();
  onProgress?.({
    step: 'chunk_summary',
    chunkIndex: window.index,
    totalChunks,
    message: `Summarizing transcript chunk ${window.index + 1}/${totalChunks}`,
  });

  const output = await llm.generate([
    {
      role: 'system',
      content: 'Summarize one transcript window. Return strict JSON only.',
    },
    {
      role: 'user',
      content: `Return a JSON array of 3 to 8 detailed points that summarize what is discussed in this transcript window.
Capture every meaningful topic, explanation, concern, option, decision, follow-up, and context from this window.
Write each point as a complete sentence with concrete detail.
Do not mention that this is a window or chunk.

${window.text}`,
    },
  ], {
    maxNewTokens: CHUNK_SUMMARY_MAX_NEW_TOKENS,
    temperature: 0.2,
  });

  return ChunkSummarySchema.parse(extractJsonArray(output));
}

async function mergeSummaryGroups(
  chunkSummaries: string[],
  onProgress?: SummaryProgress,
) {
  if (estimateTokens(chunkSummaries.join('\n\n')) <= SUMMARY_CONTEXT_TOKEN_LIMIT) return chunkSummaries;

  const llm = await loadLlmSession();
  const groups: string[][] = [];
  let current: string[] = [];
  let currentTokens = 0;

  for (const summary of chunkSummaries) {
    const nextTokens = estimateTokens(summary);
    if (current.length > 0 && currentTokens + nextTokens > 1800) {
      groups.push(current);
      current = [];
      currentTokens = 0;
    }
    current.push(summary);
    currentTokens += nextTokens;
  }
  if (current.length > 0) groups.push(current);

  const merged: string[] = [];
  for (let index = 0; index < groups.length; index++) {
    onProgress?.({
      step: 'merge_summary',
      chunkIndex: index,
      totalChunks: groups.length,
      message: `Merging transcript summary group ${index + 1}/${groups.length}`,
    });

    const output = await llm.generate([
      {
        role: 'system',
        content: 'Merge transcript summaries. Return strict JSON only.',
      },
      {
        role: 'user',
        content: `Merge these transcript summary notes into 6 to 12 detailed chronological points.
Return strict JSON only as an array of strings.
Preserve all meaningful discussion points. Do not collapse unrelated topics into one vague bullet.
Write each point as a complete sentence about what was discussed.

${groups[index].join('\n\n')}`,
      },
    ], {
      maxNewTokens: COMPLETE_SUMMARY_MAX_NEW_TOKENS,
      temperature: 0.2,
    });

    const bullets = CompleteSummarySchema.parse(extractJsonArray(output));
    merged.push(`Merged group ${index + 1}:\n${bullets.map((bullet) => `- ${bullet}`).join('\n')}`);
  }

  return mergeSummaryGroups(merged, onProgress);
}

export async function generateFinalSummary(input: {
  transcript: TranscriptTurn[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
  onProgress?: SummaryProgress;
}): Promise<string[]> {
  try {
    const llm = await loadLlmSession();
    const windows = chunkTranscriptForExtraction(input.transcript);
    const totalChunks = windows.length;
    if (totalChunks === 0) return buildFallbackSummary(input);

    const chunkSummaries = [];
    for (const window of windows) {
      const bullets = await summarizeWindow(window, totalChunks, input.onProgress);
      chunkSummaries.push(summaryLine(window, bullets));
    }

    const mergedSummaries = await mergeSummaryGroups(chunkSummaries, input.onProgress);
    const context = buildSummaryContext({
      chunkSummaries: mergedSummaries,
    });

    input.onProgress?.({
      step: 'final_summary',
      chunkIndex: totalChunks,
      totalChunks,
      message: 'Generating final whole transcript summary',
    });

    const prompt = `Create a complete meeting summary from transcript summaries that cover every part of the meeting.
Return strict JSON only as an array of 6 to 12 strings.
Each string must be a detailed summary point without markdown syntax.
Cover what was discussed across the whole transcript in chronological order.
Include the main topics, context, explanations, concerns, options, decisions, and follow-ups.
Do not summarize only extracted decisions, action items, or questions.
Do not write generic counts or say that sections were extracted.

${context}`;

    const output = await llm.generate([
      {
        role: 'system',
        content: 'You summarize whole meeting transcripts from local chunk summaries. Return strict JSON only.',
      },
      { role: 'user', content: prompt },
    ], {
      maxNewTokens: COMPLETE_SUMMARY_MAX_NEW_TOKENS,
      temperature: 0.2,
    });

    return CompleteSummarySchema.parse(extractJsonArray(output));
  } catch (error) {
    console.warn('[summarize] Summary generation failed', error);
    return buildFallbackSummary(input);
  }
}

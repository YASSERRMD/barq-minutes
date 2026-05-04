import { z } from 'zod';
import { SUMMARY_MAX_NEW_TOKENS } from '../models/modelConfig';
import { loadLlmSession } from '../models/llmSession';
import type { ActionItem, Decision, OpenQuestion, TranscriptTurn } from '../schemas/meeting';
import { estimateTokens } from '../utils/tokens';
import { chunkTranscriptForExtraction, type TranscriptWindow } from './chunker';

const SUMMARY_CONTEXT_TOKEN_LIMIT = 2800;
const CHUNK_SUMMARY_MAX_NEW_TOKENS = 256;
const SummarySchema = z.array(z.string().min(1)).length(5);
const ChunkSummarySchema = z.array(z.string().min(1)).min(1).max(5);

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

function structuredView(input: {
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}) {
  return JSON.stringify(
    {
      decisions: input.decisions,
      actionItems: input.actionItems,
      openQuestions: input.openQuestions,
    },
    null,
    2,
  );
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
  const transcriptText = input.transcript.map((turn) => turn.text).join(' ');
  const opening = trimText(transcriptText, 180);
  const duration = input.transcript.length > 0
    ? Math.round(Math.max(...input.transcript.map((turn) => turn.endSec)))
    : 0;
  const midpoint = input.transcript[Math.floor(input.transcript.length / 2)]?.text ?? '';
  const closing = input.transcript[input.transcript.length - 1]?.text ?? '';

  return [
    opening ? `Opening discussion: ${opening}` : 'The meeting transcript was captured but did not contain enough text for a detailed summary.',
    midpoint ? `Middle discussion: ${trimText(midpoint, 180)}` : `${input.decisions.length} decisions were extracted from the transcript.`,
    closing ? `Closing discussion: ${trimText(closing, 180)}` : `${input.actionItems.length} action items were extracted from the transcript.`,
    `${input.decisions.length} decisions, ${input.actionItems.length} action items, and ${input.openQuestions.length} open questions were extracted.`,
    duration > 0 ? `The transcript covers approximately ${Math.ceil(duration / 60)} minutes of discussion.` : 'Review the transcript and structured sections for verified details.',
  ];
}

function summaryLine(window: TranscriptWindow, bullets: string[]) {
  return [
    `Chunk ${window.index + 1} (${Math.round(window.startSec)}s-${Math.round(window.endSec)}s):`,
    ...bullets.map((bullet) => `- ${bullet}`),
  ].join('\n');
}

function buildSummaryContext(input: {
  chunkSummaries: string[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}) {
  const transcriptSummary = `Transcript chunk summaries:\n${input.chunkSummaries.join('\n\n')}`;
  const structured = `Structured items:\n${structuredView(input)}`;
  const context = `${transcriptSummary}\n\n${structured}`;

  if (estimateTokens(context) <= SUMMARY_CONTEXT_TOKEN_LIMIT) return context;

  const maxChars = SUMMARY_CONTEXT_TOKEN_LIMIT * 4;
  const reservedStructured = trimText(structured, 1800);
  return `${trimText(transcriptSummary, Math.max(1200, maxChars - reservedStructured.length))}\n\n${reservedStructured}`;
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
      content: `Return a JSON array of 3 to 5 concise bullets that summarize this transcript window.
Capture topics, decisions, concerns, outcomes, and follow-ups from this window.
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
        content: `Merge these transcript summary notes into exactly 5 concise bullets.
Return strict JSON only as an array of exactly 5 strings.

${groups[index].join('\n\n')}`,
      },
    ], {
      maxNewTokens: SUMMARY_MAX_NEW_TOKENS,
      temperature: 0.2,
    });

    const bullets = SummarySchema.parse(extractJsonArray(output));
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
      decisions: input.decisions,
      actionItems: input.actionItems,
      openQuestions: input.openQuestions,
    });

    input.onProgress?.({
      step: 'final_summary',
      chunkIndex: totalChunks,
      totalChunks,
      message: 'Generating final whole transcript summary',
    });

    const prompt = `Create a whole meeting summary from summaries that cover every transcript chunk.
Return strict JSON only as an array of exactly 5 strings.
Each string must be one useful bullet without markdown syntax.
The bullets must summarize the transcript as a whole, not just decisions or action items.

${context}`;

    const output = await llm.generate([
      {
        role: 'system',
        content: 'You summarize whole meeting transcripts from local chunk summaries. Return strict JSON only.',
      },
      { role: 'user', content: prompt },
    ], {
      maxNewTokens: SUMMARY_MAX_NEW_TOKENS,
      temperature: 0.2,
    });

    return SummarySchema.parse(extractJsonArray(output));
  } catch (error) {
    console.warn('[summarize] Summary generation failed', error);
    return buildFallbackSummary(input);
  }
}

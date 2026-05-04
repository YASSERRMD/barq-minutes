import { z } from 'zod';
import { loadLlmSession } from '../models/llmSession';
import type { ActionItem, Decision, OpenQuestion, TranscriptTurn } from '../schemas/meeting';
import { estimateTokens } from '../utils/tokens';
import { chunkTranscriptForExtraction, type TranscriptWindow } from './chunker';

const SUMMARY_CONTEXT_TOKEN_LIMIT = 2800;
const CHUNK_SUMMARY_MAX_NEW_TOKENS = 256;
const COMPLETE_SUMMARY_MAX_NEW_TOKENS = 384;
const SUMMARY_MAX_ITEMS = 6;
const SUMMARY_MAX_CHARS = 160;
const CompleteSummarySchema = z.array(z.string().min(1)).min(1).max(20);
const ChunkSummarySchema = z.array(z.string().min(1)).min(1).max(20);

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

function extractSummaryArray(text: string): string[] {
  const cleanText = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  try {
    const parsed = extractJsonArray(cleanText);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  } catch {
    // Fall through to markdown and numbered list parsing.
  }

  const quotedItems = [...cleanText.matchAll(/"((?:\\.|[^"\\])*)"/g)]
    .map((match) => {
      try {
        return JSON.parse(`"${match[1]}"`) as string;
      } catch {
        return match[1].replace(/\\"/g, '"');
      }
    })
    .map((item) => item.trim())
    .filter(Boolean);

  if (quotedItems.length > 0) return quotedItems;

  return cleanText
    .split('\n')
    .map((line) => line
      .replace(/^\s*(?:[-*]|\d+\.)\s+/, '')
      .replace(/^[\s"',]+|[\s"',]+$/g, '')
      .replace(/\\"/g, '"')
      .trim())
    .filter((line) => line.length > 0)
    .filter((line) => !/^```/.test(line))
    .filter((line) => !/^[\[\],]+$/.test(line))
    .slice(0, 12);
}

function trimText(text: string, maxChars: number) {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= maxChars) return clean;
  return `${clean.slice(0, Math.max(0, maxChars - 3)).trim()}...`;
}

function compactSummaryItems(items: string[], maxItems = SUMMARY_MAX_ITEMS, maxChars = SUMMARY_MAX_CHARS) {
  return items
    .map((item) => item.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .filter((item) => !/^\[?\d+s\s*-\s*\d+s\]?\.?$/i.test(item))
    .filter((item) => !/^chunk\s+\d+\b/i.test(item))
    .filter((item) => !/^transcript section\s+\d+\b/i.test(item))
    .filter((item) => !/^merged group\s+\d+\b/i.test(item))
    .filter((item) => !/^[\[\],]+$/.test(item))
    .filter((item) => !/unfinished conversation/i.test(item))
    .filter((item) => !/^here (?:is|are)\b/i.test(item))
    .filter((item) => !/^summary\b/i.test(item))
    .slice(0, maxItems)
    .map((item) => trimText(item, maxChars));
}

function transcriptText(transcript: TranscriptTurn[]) {
  return transcript
    .map((turn) => turn.text)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function splitSentences(text: string) {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 0);
}

function sentenceGroups(sentences: string[]) {
  if (sentences.length === 0) return [];
  const targetGroups = Math.min(SUMMARY_MAX_ITEMS, Math.max(1, Math.ceil(sentences.length / 10)));
  const groupSize = Math.max(1, Math.ceil(sentences.length / targetGroups));
  const groups: string[] = [];

  for (let index = 0; index < sentences.length; index += groupSize) {
    groups.push(sentences.slice(index, index + groupSize).join(' '));
  }

  return groups.slice(0, 12);
}

export function buildFallbackSummary(input: {
  transcript: TranscriptTurn[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}) {
  const text = transcriptText(input.transcript);
  const pointLimit = text.length < 500 ? 1 : text.length < 1600 ? 3 : SUMMARY_MAX_ITEMS;
  const discussionPoints = compactSummaryItems(sentenceGroups(splitSentences(text)), pointLimit, SUMMARY_MAX_CHARS);

  if (discussionPoints.length === 0) {
    return ['The meeting transcript was captured but did not contain enough text for a detailed summary.'];
  }

  return discussionPoints;
}

export function isLegacyFallbackSummary(summary: string[]) {
  const text = summary.join('\n').toLowerCase();
  return summary.some((item) => item.length > 360)
    || summary.some((item) => item.trim().endsWith('...'))
    || summary.some((item) => /^[\[\],]+$/.test(item.trim()))
    || summary.some((item) => /\\"/.test(item))
    || summary.some((item) => /unfinished conversation/i.test(item))
    || summary.some((item) => /^\d+s to \d+s:/i.test(item.trim()))
    || summary.some((item) => /^\[?\d+s\s*-\s*\d+s\]?\.?$/i.test(item.trim()))
    || text.includes('decisions were extracted from the transcript')
    || text.includes('action items were extracted from the transcript')
    || text.includes('open questions were extracted from the transcript')
    || text.includes('review the structured sections')
    || text.includes('meeting summary could not be generated')
    || text.includes('the meeting covered:')
    || text.includes('the transcript covers approximately');
}

export function displaySummary(input: {
  transcript: TranscriptTurn[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
  summary: string[];
}) {
  if (input.summary.length > 0 && !isLegacyFallbackSummary(input.summary)) {
    return input.summary;
  }

  return buildFallbackSummary(input);
}

function summaryLine(window: TranscriptWindow, bullets: string[]) {
  return [
    `Transcript section ${window.index + 1}:`,
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
      content: `Return a JSON array of 1 to 4 concise points that summarize what is discussed in this transcript window.
Capture the meaningful topics, explanations, concerns, options, decisions, follow-ups, and context from this window.
Write each point as one clear sentence under ${SUMMARY_MAX_CHARS} characters.
Do not mention that this is a window or chunk.

${window.text}`,
    },
  ], {
    maxNewTokens: CHUNK_SUMMARY_MAX_NEW_TOKENS,
    temperature: 0.2,
  });

  return compactSummaryItems(ChunkSummarySchema.parse(extractSummaryArray(output)), 4);
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
        content: `Merge these transcript summary notes into 3 to ${SUMMARY_MAX_ITEMS} concise chronological points.
Return strict JSON only as an array of strings.
Preserve the important discussion points while combining related details.
Write each point as one clear sentence under ${SUMMARY_MAX_CHARS} characters.

${groups[index].join('\n\n')}`,
      },
    ], {
      maxNewTokens: COMPLETE_SUMMARY_MAX_NEW_TOKENS,
      temperature: 0.2,
    });

    const bullets = compactSummaryItems(CompleteSummarySchema.parse(extractSummaryArray(output)));
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
  fallbackOnError?: boolean;
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

    const prompt = `Create a concise meeting summary from transcript summaries that cover every part of the meeting.
Return strict JSON only as an array of 3 to ${SUMMARY_MAX_ITEMS} strings.
Each string must be one clear summary point without markdown syntax.
Cover what was discussed across the whole transcript in chronological order.
Include the main topics, context, explanations, concerns, options, decisions, and follow-ups.
Do not summarize only extracted decisions, action items, or questions.
Do not write generic counts or say that sections were extracted.
Keep each string under ${SUMMARY_MAX_CHARS} characters.

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

    return compactSummaryItems(CompleteSummarySchema.parse(extractSummaryArray(output)));
  } catch (error) {
    console.warn('[summarize] Summary generation failed', error);
    if (input.fallbackOnError === false) throw error;
    return buildFallbackSummary(input);
  }
}

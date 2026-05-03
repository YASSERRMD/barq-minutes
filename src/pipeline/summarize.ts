import { z } from 'zod';
import { SUMMARY_MAX_NEW_TOKENS } from '../models/modelConfig';
import { loadLlmSession } from '../models/llmSession';
import type { ActionItem, Decision, OpenQuestion, TranscriptTurn } from '../schemas/meeting';
import { estimateTokens } from '../utils/tokens';

const SUMMARY_VIEW_TOKEN_LIMIT = 1500;
const SUMMARY_STRUCTURED_TOKEN_RESERVE = 900;
const SummarySchema = z.array(z.string().min(1)).length(5);

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

function speakerView(transcript: TranscriptTurn[]) {
  return transcript
    .map((turn) => `[${Math.round(turn.startSec)}s] ${turn.speaker}: ${turn.text}`)
    .join('\n');
}

function trimText(text: string, maxChars: number) {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= maxChars) return clean;
  return `${clean.slice(0, Math.max(0, maxChars - 3)).trim()}...`;
}

function transcriptEvidence(transcript: TranscriptTurn[], tokenBudget: number) {
  if (transcript.length === 0 || tokenBudget <= 0) return '';

  const maxChars = Math.max(600, tokenBudget * 4);
  const selected = new Map<number, TranscriptTurn>();
  const anchors = transcript.length <= 12
    ? transcript.map((_, index) => index)
    : [
        0,
        1,
        2,
        Math.floor(transcript.length * 0.33),
        Math.floor(transcript.length * 0.5),
        Math.floor(transcript.length * 0.66),
        transcript.length - 3,
        transcript.length - 2,
        transcript.length - 1,
      ];

  for (const index of anchors) {
    const turn = transcript[index];
    if (turn) selected.set(index, turn);
  }

  const lines: string[] = [];
  let usedChars = 0;
  for (const turn of selected.values()) {
    const line = `[${Math.round(turn.startSec)}s] ${turn.speaker}: ${trimText(turn.text, 280)}`;
    if (usedChars + line.length > maxChars) break;
    lines.push(line);
    usedChars += line.length;
  }

  return lines.join('\n');
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

  return [
    opening ? `The meeting covered: ${opening}` : 'The meeting transcript was captured but did not contain enough text for a detailed summary.',
    `${input.decisions.length} decisions were extracted from the transcript.`,
    `${input.actionItems.length} action items were extracted from the transcript.`,
    `${input.openQuestions.length} open questions were extracted from the transcript.`,
    duration > 0 ? `The transcript covers approximately ${Math.ceil(duration / 60)} minutes of discussion.` : 'Review the transcript and structured sections for verified details.',
  ];
}

export function buildSummaryContext(input: {
  transcript: TranscriptTurn[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}) {
  const structured = `Structured items:\n${structuredView(input)}`;
  const withSpeakers = `${structured}\n\nSpeaker turns:\n${speakerView(input.transcript)}`;
  if (estimateTokens(withSpeakers) <= SUMMARY_VIEW_TOKEN_LIMIT) return withSpeakers;

  const structuredTokens = estimateTokens(structured);
  const evidenceBudget = Math.max(0, SUMMARY_VIEW_TOKEN_LIMIT - Math.min(structuredTokens, SUMMARY_STRUCTURED_TOKEN_RESERVE));
  const evidence = transcriptEvidence(input.transcript, evidenceBudget);
  if (!evidence) return structured;

  const compact = `${structured}\n\nTranscript evidence:\n${evidence}`;
  if (estimateTokens(compact) <= SUMMARY_VIEW_TOKEN_LIMIT) return compact;
  return `${structured}\n\nTranscript evidence:\n${transcriptEvidence(input.transcript, 350)}`;
}

export async function generateFinalSummary(input: {
  transcript: TranscriptTurn[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}): Promise<string[]> {
  try {
    const llm = await loadLlmSession();
    const context = buildSummaryContext(input);
    const prompt = `Create a concise executive summary from the compact meeting view.
Return strict JSON only as an array of exactly 5 strings.
Each string must be one useful bullet without markdown syntax.

${context}`;

    const output = await llm.generate([
      {
        role: 'system',
        content: 'You summarize meetings from structured local data. Return strict JSON only.',
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

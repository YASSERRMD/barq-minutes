import { z } from 'zod';
import { SUMMARY_MAX_NEW_TOKENS } from '../models/modelConfig';
import { loadLlmSession } from '../models/llmSession';
import type { ActionItem, Decision, OpenQuestion, TranscriptTurn } from '../schemas/meeting';
import { estimateTokens } from '../utils/tokens';

const SUMMARY_VIEW_TOKEN_LIMIT = 1500;
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

export function buildSummaryContext(input: {
  transcript: TranscriptTurn[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}) {
  const structured = `Structured items:\n${structuredView(input)}`;
  const withSpeakers = `${structured}\n\nSpeaker turns:\n${speakerView(input.transcript)}`;
  if (estimateTokens(withSpeakers) <= SUMMARY_VIEW_TOKEN_LIMIT) return withSpeakers;
  return structured;
}

export async function generateFinalSummary(input: {
  transcript: TranscriptTurn[];
  decisions: Decision[];
  actionItems: ActionItem[];
  openQuestions: OpenQuestion[];
}): Promise<string[]> {
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

  try {
    return SummarySchema.parse(extractJsonArray(output));
  } catch (error) {
    console.warn('[summarize] Invalid summary JSON', error);
    return [
      'The meeting summary could not be generated from the local model output.',
      `${input.decisions.length} decisions were extracted.`,
      `${input.actionItems.length} action items were extracted.`,
      `${input.openQuestions.length} open questions were extracted.`,
      'Review the structured sections for verified details.',
    ];
  }
}

import { z } from 'zod';
import { DecisionSchema, type Decision } from '../schemas/meeting';
import type { TranscriptWindow } from './chunker';
import { runJsonExtraction, type ExtractionProgress } from './extractJson';

const DecisionArraySchema = z.array(DecisionSchema);

const DECISION_SYSTEM_PROMPT = `Extract explicit decisions from this transcript window.
Return only a JSON array matching this TypeScript shape:
[{"text":"decision made","rationale":null,"speaker":null,"timestampSec":0}]
Rules:
- Include only decisions that were agreed, approved, rejected, selected, or finalized.
- Do not include action items or open questions.
- Use null for unknown rationale or speaker.
- Use the closest timestamp in seconds from the transcript.
- Return [] if there are no decisions.`;

const DECISION_RETRY_PROMPT = `${DECISION_SYSTEM_PROMPT}
Your last response was invalid. Return strict JSON only. No prose, no markdown, no trailing commas.`;

export async function extractDecisions(
  window: TranscriptWindow,
  totalChunks = 1,
  onProgress?: ExtractionProgress,
): Promise<Decision[]> {
  return runJsonExtraction({
    step: 'extract_decisions',
    schemaName: 'Decision[]',
    schema: DecisionArraySchema,
    systemPrompt: DECISION_SYSTEM_PROMPT,
    retryPrompt: DECISION_RETRY_PROMPT,
    windowText: window.text,
    chunkIndex: window.index,
    totalChunks,
    onProgress,
  });
}

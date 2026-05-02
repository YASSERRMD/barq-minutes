import { z } from 'zod';
import { ActionItemSchema, type ActionItem } from '../schemas/meeting';
import type { TranscriptWindow } from './chunker';
import { runJsonExtraction, type ExtractionProgress } from './extractJson';

const ActionItemArraySchema = z.array(ActionItemSchema);

const ACTION_SYSTEM_PROMPT = `Extract action items from this transcript window.
Return only a JSON array matching this TypeScript shape:
[{"text":"task to complete","owner":null,"dueDate":null,"timestampSec":0}]
Rules:
- Include only concrete tasks, follow-ups, assignments, or commitments.
- Do not include decisions or open questions.
- Use null for unknown owner.
- dueDate must be an ISO date string if a date is explicitly stated, otherwise null.
- Use the closest timestamp in seconds from the transcript.
- Return [] if there are no action items.`;

const ACTION_RETRY_PROMPT = `${ACTION_SYSTEM_PROMPT}
Your last response was invalid. Return strict JSON only. No prose, no markdown, no trailing commas.`;

export async function extractActionItems(
  window: TranscriptWindow,
  totalChunks = 1,
  onProgress?: ExtractionProgress,
): Promise<ActionItem[]> {
  return runJsonExtraction({
    step: 'extract_action_items',
    schemaName: 'ActionItem[]',
    schema: ActionItemArraySchema,
    systemPrompt: ACTION_SYSTEM_PROMPT,
    retryPrompt: ACTION_RETRY_PROMPT,
    windowText: window.text,
    chunkIndex: window.index,
    totalChunks,
    onProgress,
  });
}

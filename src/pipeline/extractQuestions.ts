import { z } from 'zod';
import { OpenQuestionSchema, type OpenQuestion } from '../schemas/meeting';
import type { TranscriptWindow } from './chunker';
import { runJsonExtraction, type ExtractionProgress } from './extractJson';

const OpenQuestionArraySchema = z.array(OpenQuestionSchema);

const QUESTION_SYSTEM_PROMPT = `Extract unresolved open questions from this transcript window.
Return only a JSON array matching this TypeScript shape:
[{"text":"question that remains open","raisedBy":null,"timestampSec":0}]
Rules:
- Include only questions that are unresolved or need later follow-up.
- Do not include answered clarifying questions.
- Do not include decisions or action items.
- Use null for unknown raisedBy.
- Use the closest timestamp in seconds from the transcript.
- Return [] if there are no open questions.`;

const QUESTION_RETRY_PROMPT = `${QUESTION_SYSTEM_PROMPT}
Your last response was invalid. Return strict JSON only. No prose, no markdown, no trailing commas.`;

export async function extractQuestions(
  window: TranscriptWindow,
  totalChunks = 1,
  onProgress?: ExtractionProgress,
): Promise<OpenQuestion[]> {
  return runJsonExtraction({
    step: 'extract_questions',
    schemaName: 'OpenQuestion[]',
    schema: OpenQuestionArraySchema,
    systemPrompt: QUESTION_SYSTEM_PROMPT,
    retryPrompt: QUESTION_RETRY_PROMPT,
    windowText: window.text,
    chunkIndex: window.index,
    totalChunks,
    onProgress,
  });
}

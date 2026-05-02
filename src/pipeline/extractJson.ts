import { z } from 'zod';
import { STRUCTURED_MAX_NEW_TOKENS } from '../models/modelConfig';
import { loadLlmSession, type ChatMessage } from '../models/llmSession';

export type ExtractionProgress = (event: {
  step: string;
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
    throw new Error('Model did not return a JSON array');
  }
  return JSON.parse(candidate.slice(start, end + 1));
}

export async function runJsonExtraction<T>(
  input: {
    step: string;
    schemaName: string;
    schema: z.ZodType<T[]>;
    systemPrompt: string;
    retryPrompt: string;
    windowText: string;
    chunkIndex: number;
    totalChunks: number;
    onProgress?: ExtractionProgress;
  },
): Promise<T[]> {
  const llm = await loadLlmSession();

  const messages: ChatMessage[] = [
    { role: 'system', content: input.systemPrompt },
    { role: 'user', content: input.windowText },
  ];

  input.onProgress?.({
    step: input.step,
    chunkIndex: input.chunkIndex,
    totalChunks: input.totalChunks,
    message: `Processing chunk ${input.chunkIndex + 1}/${input.totalChunks}`,
  });

  for (let attempt = 0; attempt < 2; attempt++) {
    const output = await llm.generate(attempt === 0 ? messages : [
      { role: 'system', content: input.retryPrompt },
      { role: 'user', content: input.windowText },
    ], {
      maxNewTokens: STRUCTURED_MAX_NEW_TOKENS,
      temperature: 0.1,
    });

    try {
      const parsed = extractJsonArray(output);
      return input.schema.parse(parsed);
    } catch (error) {
      if (attempt === 1) {
        console.warn(`[${input.step}] ${input.schemaName} validation failed`, error);
        return [];
      }
    }
  }

  return [];
}

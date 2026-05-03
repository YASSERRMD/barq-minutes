import { loadLlmSession } from '../models/llmSession';
import type { Meeting } from '../schemas/meeting';
import {
  indexTranscriptChunks,
  searchMeetingChunks,
  type RetrievedChunk,
  type TranscriptChunk,
} from '../storage/vector';
import { chunkTranscriptForRag } from './chunker';

const QA_MAX_CONTEXT_CHARS = 7000;

export type AskMeetingResult = {
  answer: string;
  chunks: RetrievedChunk[];
};

function toTranscriptChunks(meetingId: string, windows: ReturnType<typeof chunkTranscriptForRag>): TranscriptChunk[] {
  return windows.map((window) => ({
    id: `${meetingId}:rag:${window.index}`,
    meetingId,
    text: window.text,
    startSec: window.startSec,
    endSec: window.endSec,
    turnIndexes: window.turnIndexes,
  }));
}

export async function indexMeetingForAsk(meeting: Meeting): Promise<number> {
  const windows = chunkTranscriptForRag(meeting.transcript, 512, 64);
  return indexTranscriptChunks(toTranscriptChunks(meeting.id, windows));
}

function buildContext(chunks: RetrievedChunk[]) {
  return chunks
    .map((chunk, index) => {
      const header = `[${index + 1}] ${Math.round(chunk.startSec)}s-${Math.round(chunk.endSec)}s`;
      return `${header}\n${chunk.text}`;
    })
    .join('\n\n')
    .slice(0, QA_MAX_CONTEXT_CHARS);
}

export async function askMeeting(meetingId: string, question: string): Promise<AskMeetingResult> {
  const chunks = await searchMeetingChunks(meetingId, question, 5);
  const context = buildContext(chunks);
  const llm = await loadLlmSession();

  const answer = await llm.generate([
    {
      role: 'system',
      content: `Answer questions about one meeting using only the retrieved transcript chunks.
If the chunks do not contain the answer, say that the meeting transcript does not show it.
Keep the answer concise and cite chunk numbers in square brackets.
Do not include thinking, hidden reasoning, or <think> tags.`,
    },
    {
      role: 'user',
      content: `Question: ${question}

Retrieved chunks:
${context}`,
    },
  ], {
    maxNewTokens: 384,
    temperature: 0.1,
  });

  return { answer, chunks };
}

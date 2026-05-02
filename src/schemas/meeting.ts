import { z } from 'zod';

export const TranscriptTurnSchema = z.object({
  speaker: z.string().min(1),
  text: z.string(),
  startSec: z.number().nonnegative(),
  endSec: z.number().nonnegative(),
});

export const DecisionSchema = z.object({
  text: z.string().min(1),
  rationale: z.string().nullable(),
  speaker: z.string().nullable(),
  timestampSec: z.number().nonnegative(),
});

export const ActionItemSchema = z.object({
  text: z.string().min(1),
  owner: z.string().nullable(),
  dueDate: z.string().nullable(),
  timestampSec: z.number().nonnegative(),
});

export const OpenQuestionSchema = z.object({
  text: z.string().min(1),
  raisedBy: z.string().nullable(),
  timestampSec: z.number().nonnegative(),
});

export const MeetingSchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1),
  startedAt: z.number().int(),
  endedAt: z.number().int(),
  durationSec: z.number().nonnegative(),
  participants: z.array(z.string()),
  audioBlobKey: z.string().nullable(),
  transcript: z.array(TranscriptTurnSchema),
  decisions: z.array(DecisionSchema),
  actionItems: z.array(ActionItemSchema),
  openQuestions: z.array(OpenQuestionSchema),
  summary: z.array(z.string()),
  tags: z.array(z.string()),
});

export type TranscriptTurn = z.infer<typeof TranscriptTurnSchema>;
export type Decision = z.infer<typeof DecisionSchema>;
export type ActionItem = z.infer<typeof ActionItemSchema>;
export type OpenQuestion = z.infer<typeof OpenQuestionSchema>;
export type Meeting = z.infer<typeof MeetingSchema>;

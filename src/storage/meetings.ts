import { del, entries, get, set } from 'idb-keyval';
import { MeetingSchema, type Meeting } from '../schemas/meeting';

const MEETING_PREFIX = 'meeting:';

function meetingKey(id: string) {
  return `${MEETING_PREFIX}${id}`;
}

export async function saveMeeting(meeting: Meeting): Promise<Meeting> {
  const parsed = MeetingSchema.parse(meeting);
  await set(meetingKey(parsed.id), parsed);
  return parsed;
}

export async function getMeeting(id: string): Promise<Meeting | null> {
  const value = await get(meetingKey(id));
  if (!value) return null;
  return MeetingSchema.parse(value);
}

export async function listMeetings(): Promise<Meeting[]> {
  const allEntries = await entries();
  return allEntries
    .filter(([key]) => typeof key === 'string' && key.startsWith(MEETING_PREFIX))
    .map(([, value]) => MeetingSchema.parse(value))
    .sort((a, b) => b.startedAt - a.startedAt);
}

export async function deleteMeeting(id: string): Promise<void> {
  await del(meetingKey(id));
}

export async function clearMeetings(): Promise<void> {
  const allEntries = await entries();
  await Promise.all(
    allEntries
      .filter(([key]) => typeof key === 'string' && key.startsWith(MEETING_PREFIX))
      .map(([key]) => del(key)),
  );
}

export function createEmptyMeeting(input: {
  id: string;
  title: string;
  startedAt: number;
  endedAt: number;
  durationSec: number;
  audioBlobKey?: string | null;
}): Meeting {
  return {
    id: input.id,
    title: input.title.trim() || 'Untitled meeting',
    startedAt: input.startedAt,
    endedAt: input.endedAt,
    durationSec: input.durationSec,
    participants: [],
    audioBlobKey: input.audioBlobKey ?? null,
    transcript: [],
    decisions: [],
    actionItems: [],
    openQuestions: [],
    summary: [],
    tags: [],
  };
}

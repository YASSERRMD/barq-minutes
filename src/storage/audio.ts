import { del, entries, get, set } from 'idb-keyval';

const AUDIO_PREFIX = 'audio:';

export function audioKey(meetingId: string) {
  return `${AUDIO_PREFIX}${meetingId}`;
}

export async function saveAudioBlob(meetingId: string, blob: Blob): Promise<string> {
  const key = audioKey(meetingId);
  await set(key, blob);
  return key;
}

export async function getAudioBlob(key: string): Promise<Blob | null> {
  return (await get(key)) ?? null;
}

export async function deleteAudioBlob(key: string): Promise<void> {
  await del(key);
}

export async function clearAudioBlobs(): Promise<void> {
  const allEntries = await entries();
  await Promise.all(
    allEntries
      .filter(([key]) => typeof key === 'string' && key.startsWith(AUDIO_PREFIX))
      .map(([key]) => del(key)),
  );
}

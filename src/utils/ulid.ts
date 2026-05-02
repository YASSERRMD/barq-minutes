const ENCODING = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

function encodeTime(timeMs: number, length: number) {
  let value = Math.floor(timeMs);
  let output = '';
  for (let i = length - 1; i >= 0; i--) {
    output = ENCODING[value % 32] + output;
    value = Math.floor(value / 32);
  }
  return output;
}

function encodeRandom(length: number) {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => ENCODING[byte % 32]).join('');
}

export function ulid(now = Date.now()): string {
  return `${encodeTime(now, 10)}${encodeRandom(16)}`;
}

import type { TranscriptTurn } from '../schemas/meeting';
import { formatClock } from '../utils/time';

export default function TranscriptViewer({ transcript }: { transcript: TranscriptTurn[] }) {
  if (transcript.length === 0) {
    return <div className="empty-state">No transcript turns captured</div>;
  }

  return (
    <div className="transcript-list">
      {transcript.map((turn, index) => (
        <article key={`${turn.startSec}-${index}`} className="transcript-turn">
          <time>{formatClock(turn.startSec)}</time>
          <div>
            <h3>{turn.speaker}</h3>
            <p>{turn.text}</p>
          </div>
        </article>
      ))}
    </div>
  );
}

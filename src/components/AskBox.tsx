import { useState } from 'react';
import { Send, Search } from 'lucide-react';
import { askMeeting, type AskMeetingResult } from '../pipeline/askMeeting';
import { formatClock } from '../utils/time';

export default function AskBox({ meetingId }: { meetingId: string }) {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<AskMeetingResult | null>(null);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isAsking) return;

    setIsAsking(true);
    setError(null);
    try {
      setResult(await askMeeting(meetingId, trimmed));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <div className="ask-box">
      <form className="ask-form" onSubmit={submit}>
        <Search size={18} />
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask this meeting"
          aria-label="Ask this meeting"
        />
        <button className="icon-button" type="submit" disabled={isAsking} aria-label="Ask">
          <Send size={17} />
        </button>
      </form>

      {isAsking ? <p className="status-line">Retrieving local chunks and loading the LLM</p> : null}
      {error ? <p className="status-line">{error}</p> : null}
      {result ? (
        <div className="ask-result">
          <p>{result.answer}</p>
          <div className="source-grid">
            {result.chunks.map((chunk, index) => (
              <article key={chunk.id} className="source-card">
                <strong>
                  [{index + 1}] {formatClock(chunk.startSec)} to {formatClock(chunk.endSec)}
                </strong>
                <p>{chunk.text}</p>
              </article>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

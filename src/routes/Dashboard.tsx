import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Mic, Search } from 'lucide-react';
import MeetingCard from '../components/MeetingCard';
import type { Meeting } from '../schemas/meeting';
import { listMeetings } from '../storage/meetings';

export default function Dashboard() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    listMeetings()
      .then((items) => {
        if (mounted) setMeetings(items);
      })
      .finally(() => {
        if (mounted) setIsLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return meetings;
    return meetings.filter((meeting) => {
      const haystack = [
        meeting.title,
        meeting.tags.join(' '),
        meeting.participants.join(' '),
        meeting.summary.join(' '),
        meeting.decisions.map((item) => item.text).join(' '),
        meeting.actionItems.map((item) => item.text).join(' '),
      ].join(' ').toLowerCase();
      return haystack.includes(normalized);
    });
  }, [meetings, query]);

  return (
    <section>
      <section className="app-banner" aria-label="barq-minutes overview">
        <div className="app-banner-copy">
          <p className="page-kicker">Private meeting intelligence</p>
          <h1>barq-minutes</h1>
          <p>Record, transcribe, extract decisions, and search meetings locally in the browser.</p>
        </div>
        <img src="/readme-hero.png" alt="barq-minutes local meeting intelligence workflow" />
      </section>

      <header className="page-header">
        <div>
          <p className="page-kicker">Dashboard</p>
          <h1 className="page-title">Meetings</h1>
          <p className="page-subtitle">
            Search private transcripts, decisions, action items, and summaries stored in IndexedDB.
          </p>
        </div>
        <Link className="button primary" to="/record">
          <Mic size={18} />
          New Meeting
        </Link>
      </header>

      <div className="toolbar panel">
        <label className="search-field">
          <Search size={18} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search meetings"
            aria-label="Search meetings"
          />
        </label>
      </div>

      {isLoading ? (
        <div className="empty-state">Loading meetings</div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <div>
            <h2>No meetings found</h2>
            <p>Record a meeting or adjust the search text.</p>
          </div>
        </div>
      ) : (
        <div className="meeting-grid">
          {filtered.map((meeting) => (
            <MeetingCard key={meeting.id} meeting={meeting} />
          ))}
        </div>
      )}
    </section>
  );
}

import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, Calendar, Clock } from 'lucide-react';
import AskBox from '../components/AskBox';
import ExportButtons from '../components/ExportButtons';
import TranscriptViewer from '../components/TranscriptViewer';
import type { Meeting as MeetingRecord } from '../schemas/meeting';
import { getMeeting } from '../storage/meetings';
import { formatClock, formatDateTime } from '../utils/time';

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel detail-section">
      <h2 className="section-title">{title}</h2>
      {children}
    </section>
  );
}

export default function Meeting() {
  const { id } = useParams();
  const [meeting, setMeeting] = useState<MeetingRecord | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    let mounted = true;
    getMeeting(id)
      .then((record) => {
        if (mounted) setMeeting(record);
      })
      .finally(() => {
        if (mounted) setIsLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [id]);

  if (isLoading) return <div className="empty-state">Loading meeting</div>;

  if (!meeting) {
    return (
      <div className="empty-state">
        <div>
          <h2>Meeting not found</h2>
          <Link className="button" to="/">
            <ArrowLeft size={16} />
            Back
          </Link>
        </div>
      </div>
    );
  }

  return (
    <section>
      <header className="page-header">
        <div>
          <Link className="back-link" to="/">
            <ArrowLeft size={16} />
            Meetings
          </Link>
          <h1 className="page-title">{meeting.title}</h1>
          <div className="detail-meta">
            <span>
              <Calendar size={16} />
              {formatDateTime(meeting.startedAt)}
            </span>
            <span>
              <Clock size={16} />
              {formatClock(meeting.durationSec)}
            </span>
          </div>
        </div>
        <ExportButtons meeting={meeting} />
      </header>

      <div className="detail-grid">
        <Section title="Ask This Meeting">
          <AskBox meetingId={meeting.id} />
        </Section>

        <Section title="Executive Summary">
          <ul className="summary-list">
            {meeting.summary.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
            {meeting.summary.length === 0 ? <li>No summary generated</li> : null}
          </ul>
        </Section>

        <Section title="Decisions">
          <div className="item-list">
            {meeting.decisions.map((item, index) => (
              <article key={index} className="structured-item">
                <p>{item.text}</p>
                <span>{formatClock(item.timestampSec)} {item.speaker ? `by ${item.speaker}` : ''}</span>
              </article>
            ))}
            {meeting.decisions.length === 0 ? <p className="muted">No decisions extracted</p> : null}
          </div>
        </Section>

        <Section title="Action Items">
          <div className="item-list">
            {meeting.actionItems.map((item, index) => (
              <article key={index} className="structured-item">
                <p>{item.text}</p>
                <span>
                  {formatClock(item.timestampSec)}
                  {item.owner ? ` for ${item.owner}` : ''}
                  {item.dueDate ? ` due ${item.dueDate}` : ''}
                </span>
              </article>
            ))}
            {meeting.actionItems.length === 0 ? <p className="muted">No action items extracted</p> : null}
          </div>
        </Section>

        <Section title="Open Questions">
          <div className="item-list">
            {meeting.openQuestions.map((item, index) => (
              <article key={index} className="structured-item">
                <p>{item.text}</p>
                <span>{formatClock(item.timestampSec)} {item.raisedBy ? `raised by ${item.raisedBy}` : ''}</span>
              </article>
            ))}
            {meeting.openQuestions.length === 0 ? <p className="muted">No open questions extracted</p> : null}
          </div>
        </Section>

        <Section title="Transcript">
          <TranscriptViewer transcript={meeting.transcript} />
        </Section>
      </div>
    </section>
  );
}

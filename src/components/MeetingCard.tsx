import { Link } from 'react-router-dom';
import { ArrowRight, Clock, Users } from 'lucide-react';
import type { Meeting } from '../schemas/meeting';
import { formatClock, formatDateTime } from '../utils/time';

export default function MeetingCard({ meeting }: { meeting: Meeting }) {
  return (
    <article className="meeting-card card">
      <div>
        <p className="meeting-card-date">{formatDateTime(meeting.startedAt)}</p>
        <h2>{meeting.title}</h2>
      </div>
      <div className="meeting-card-meta">
        <span>
          <Clock size={16} />
          {formatClock(meeting.durationSec)}
        </span>
        <span>
          <Users size={16} />
          {meeting.participants.length || 'Unknown'}
        </span>
      </div>
      <p className="meeting-card-summary">
        {meeting.summary[0] ?? `${meeting.transcript.length} transcript turns captured`}
      </p>
      <div className="meeting-card-counts">
        <span>{meeting.decisions.length} decisions</span>
        <span>{meeting.actionItems.length} actions</span>
        <span>{meeting.openQuestions.length} questions</span>
      </div>
      <Link className="button" to={`/meeting/${meeting.id}`}>
        Open
        <ArrowRight size={16} />
      </Link>
    </article>
  );
}

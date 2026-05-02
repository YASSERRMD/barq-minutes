export type ProgressStep = {
  id: string;
  label: string;
  detail: string;
  status: 'pending' | 'active' | 'done';
  progress?: number | null;
};

export default function ProgressTimeline({ steps }: { steps: ProgressStep[] }) {
  return (
    <ol className="progress-timeline" aria-label="Processing progress">
      {steps.map((step) => (
        <li key={step.id} className={`progress-step ${step.status}`}>
          <div className="progress-dot" />
          <div className="progress-content">
            <div className="progress-row">
              <span>{step.label}</span>
              <span>{step.status}</span>
            </div>
            <p>{step.detail}</p>
            {typeof step.progress === 'number' ? (
              <div className="progress-bar" aria-label={`${step.label} progress`}>
                <span style={{ width: `${Math.max(0, Math.min(100, step.progress))}%` }} />
              </div>
            ) : null}
          </div>
        </li>
      ))}
    </ol>
  );
}

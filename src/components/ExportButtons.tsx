import { Download, FileText } from 'lucide-react';
import { toPng } from 'html-to-image';
import jsPDF from 'jspdf';
import { buildFallbackSummary } from '../pipeline/summarize';
import type { Meeting } from '../schemas/meeting';
import { formatClock, formatDateTime } from '../utils/time';

function markdown(meeting: Meeting) {
  const summary = meeting.summary.length > 0 ? meeting.summary : buildFallbackSummary(meeting);
  const lines = [
    `# ${meeting.title}`,
    '',
    `Started: ${formatDateTime(meeting.startedAt)}`,
    `Duration: ${formatClock(meeting.durationSec)}`,
    '',
    '## Summary',
    ...summary.map((item) => `- ${item}`),
    '',
    '## Decisions',
    ...meeting.decisions.map((item) => `- ${formatClock(item.timestampSec)} ${item.text}`),
    '',
    '## Action Items',
    ...meeting.actionItems.map((item) => `- ${formatClock(item.timestampSec)} ${item.text}${item.owner ? ` Owner: ${item.owner}` : ''}${item.dueDate ? ` Due: ${item.dueDate}` : ''}`),
    '',
    '## Open Questions',
    ...meeting.openQuestions.map((item) => `- ${formatClock(item.timestampSec)} ${item.text}`),
    '',
    '## Transcript',
    ...meeting.transcript.map((turn) => `- ${formatClock(turn.startSec)} ${turn.speaker}: ${turn.text}`),
  ];
  return lines.join('\n');
}

function downloadText(filename: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/markdown;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function exportNode(meeting: Meeting) {
  const summary = meeting.summary.length > 0 ? meeting.summary : buildFallbackSummary(meeting);
  const node = document.createElement('div');
  node.style.position = 'fixed';
  node.style.left = '-10000px';
  node.style.top = '0';
  node.style.width = '900px';
  node.style.padding = '40px';
  node.style.background = 'white';
  node.style.color = 'black';
  node.style.fontFamily = 'system-ui, sans-serif';
  node.innerHTML = `
    <h1>${meeting.title}</h1>
    <p>${formatDateTime(meeting.startedAt)} | ${formatClock(meeting.durationSec)}</p>
    <h2>Summary</h2>
    <ul>${summary.map((item) => `<li>${item}</li>`).join('')}</ul>
    <h2>Decisions</h2>
    <ul>${meeting.decisions.map((item) => `<li>${item.text}</li>`).join('')}</ul>
    <h2>Action Items</h2>
    <ul>${meeting.actionItems.map((item) => `<li>${item.text}</li>`).join('')}</ul>
    <h2>Open Questions</h2>
    <ul>${meeting.openQuestions.map((item) => `<li>${item.text}</li>`).join('')}</ul>
  `;
  document.body.appendChild(node);
  return node;
}

export default function ExportButtons({ meeting }: { meeting: Meeting }) {
  async function exportPdf() {
    const node = exportNode(meeting);
    try {
      const png = await toPng(node, { pixelRatio: 2 });
      const pdf = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
      const pageWidth = pdf.internal.pageSize.getWidth();
      const imageProps = pdf.getImageProperties(png);
      const imageHeight = (imageProps.height * pageWidth) / imageProps.width;
      pdf.addImage(png, 'PNG', 0, 0, pageWidth, imageHeight);
      pdf.save(`${meeting.title.replace(/[^a-z0-9]+/gi, '-').toLowerCase() || 'meeting'}.pdf`);
    } finally {
      node.remove();
    }
  }

  return (
    <div className="export-buttons">
      <button className="button" type="button" onClick={() => downloadText(`${meeting.title}.md`, markdown(meeting))}>
        <FileText size={17} />
        Markdown
      </button>
      <button className="button" type="button" onClick={exportPdf}>
        <Download size={17} />
        PDF
      </button>
    </div>
  );
}

import { StatusData } from '../types';

interface StatusIndicatorProps {
  status: StatusData | null;
}

const TOOL_LABELS: Record<string, string> = {
  generate_quiz_question: 'Generating quiz question...',
  grade_answer: 'Grading answer...',
  search_knowledge_base: 'Searching tutor knowledge base...',
  calculate_score: 'Calculating student progress...',
};

export default function StatusIndicator({ status }: StatusIndicatorProps) {
  if (!status) return null;

  const label = TOOL_LABELS[status.tool] || `Running tool: ${status.tool}...`;

  return (
    <div className="status-indicator-container">
      <div className="status-indicator">
        <div className="status-spinner" />
        <span>{label}</span>
      </div>
    </div>
  );
}

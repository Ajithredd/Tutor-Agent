import { GradeBadgeData } from '../types';

interface GradeBadgeProps {
  data: GradeBadgeData | string;
}

export default function GradeBadge({ data }: GradeBadgeProps) {
  const gradeData: GradeBadgeData = typeof data === 'string' ? JSON.parse(data) : data;
  const isCorrect = gradeData?.correct ?? gradeData?.is_correct ?? false;
  const feedback = gradeData?.feedback || gradeData?.explanation || (isCorrect ? 'Correct answer!' : 'Incorrect.');

  return (
    <div className={`grade-badge ${isCorrect ? 'correct' : 'incorrect'}`}>
      <div className="grade-icon">{isCorrect ? '✓' : '✗'}</div>
      <div className="grade-feedback">
        <strong>{isCorrect ? 'Correct!' : 'Needs Improvement'}</strong>: {feedback}
      </div>
    </div>
  );
}

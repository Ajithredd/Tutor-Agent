import { useState } from 'react';
import { QuizQuestionData } from '../types';

interface QuizCardProps {
  data: QuizQuestionData | string;
  onSelectAnswer?: (choice: string) => void;
}

export default function QuizCard({ data, onSelectAnswer }: QuizCardProps) {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const quizData: QuizQuestionData = typeof data === 'string' ? JSON.parse(data) : data;
  const { question, options = {} } = quizData || {};

  const handleSelect = (key: string) => {
    if (selectedOption !== null) return;
    setSelectedOption(key);
    if (onSelectAnswer) {
      onSelectAnswer(key);
    }
  };

  return (
    <div className="quiz-card">
      <h3>{question || 'Quiz Question'}</h3>
      <div className="quiz-options">
        {Object.entries(options).map(([key, text]) => (
          <button
            key={key}
            className={`quiz-option-btn ${selectedOption === key ? 'selected' : ''}`}
            disabled={selectedOption !== null}
            onClick={() => handleSelect(key)}
          >
            <span className="option-key">{key}</span>
            <span>{text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

import React, { useState } from 'react';

interface ComposerProps {
  onSendMessage: (text: string) => void;
  disabled: boolean;
}

export default function Composer({ onSendMessage, disabled }: ComposerProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || disabled) return;
    onSendMessage(input.trim());
    setInput('');
  };

  return (
    <div className="composer-container">
      <form onSubmit={handleSubmit} className="composer-form">
        <input
          type="text"
          className="composer-input"
          placeholder={disabled ? 'Assistant is replying...' : 'Type your question or answer...'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={disabled}
        />
        <button
          type="submit"
          className="composer-submit"
          disabled={disabled || !input.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
}

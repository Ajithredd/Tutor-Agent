import React, { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import QuizCard from './QuizCard';
import GradeBadge from './GradeBadge';
import { Message, ToolResult, InterruptData } from '../types';

interface ChatWindowProps {
  messages: Message[];
  activeMessage: Message | null;
  onSelectQuizAnswer?: (choice: string) => void;
  onApproveInterrupt?: (approved: boolean) => void;
}

export default function ChatWindow({
  messages,
  activeMessage,
  onSelectQuizAnswer,
  onApproveInterrupt,
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeMessage]);

  const renderGenerativeUI = (toolResult: ToolResult) => {
    if (!toolResult) return null;
    const { tool, output } = toolResult;

    if (tool === 'generate_quiz_question') {
      return (
        <div className="generative-ui-container">
          <QuizCard
            data={output}
            onSelectAnswer={(choice) => onSelectQuizAnswer && onSelectQuizAnswer(choice)}
          />
        </div>
      );
    }

    if (tool === 'grade_answer') {
      return (
        <div className="generative-ui-container">
          <GradeBadge data={output} />
        </div>
      );
    }

    return null;
  };

  const renderInterruptUI = (interruptData: InterruptData | null | undefined) => {
    if (!interruptData) return null;

    return (
      <div className="approval-card">
        <div><strong>Human-In-The-Loop Approval Required:</strong></div>
        <div>{interruptData.action || interruptData.prompt || 'The agent requires approval to proceed.'}</div>
        <div className="approval-actions">
          <button
            className="approval-btn approve"
            onClick={() => onApproveInterrupt && onApproveInterrupt(true)}
          >
            Approve
          </button>
          <button
            className="approval-btn reject"
            onClick={() => onApproveInterrupt && onApproveInterrupt(false)}
          >
            Reject
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="chat-window">
      {messages.filter(Boolean).map((msg, index) => (
        <div key={index} className={`message-bubble-wrapper ${msg.role || 'assistant'}`}>
          {msg.text && (
            <div className="message-bubble">
              <MessageBubble content={msg.text} />
            </div>
          )}
          {msg.toolResults && msg.toolResults.map((tr, i) => (
            <React.Fragment key={i}>
              {renderGenerativeUI(tr)}
            </React.Fragment>
          ))}
          {msg.interrupt && renderInterruptUI(msg.interrupt)}
        </div>
      ))}

      {/* Render active streaming message buffer */}
      {activeMessage && (
        <div className="message-bubble-wrapper assistant">
          {activeMessage.text && (
            <div className="message-bubble">
              <MessageBubble content={activeMessage.text} />
            </div>
          )}
          {activeMessage.toolResults && activeMessage.toolResults.map((tr, i) => (
            <React.Fragment key={i}>
              {renderGenerativeUI(tr)}
            </React.Fragment>
          ))}
          {activeMessage.interrupt && renderInterruptUI(activeMessage.interrupt)}
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

import { useState, useEffect, useRef } from 'react';
import ChatWindow from './components/ChatWindow';
import StatusIndicator from './components/StatusIndicator';
import Composer from './components/Composer';
import { Message, StatusData, InterruptData } from './types';

function getThreadId(): string {
  let threadId = localStorage.getItem('tutor_agent_thread_id');
  if (!threadId) {
    threadId = crypto.randomUUID();
    localStorage.setItem('tutor_agent_thread_id', threadId);
  }
  return threadId;
}

export default function App() {
  const [threadId] = useState<string>(getThreadId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeMessage, setActiveMessage] = useState<Message | null>(null);
  const [currentStatus, setCurrentStatus] = useState<StatusData | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  const messageQueue = useRef<string[]>([]);
  const activeMessageRef = useRef<Message | null>(null);

  useEffect(() => {
    activeMessageRef.current = activeMessage;
  }, [activeMessage]);

  const sendRequest = async (userText: string) => {
    setIsStreaming(true);
    setCurrentStatus(null);

    const initialActive: Message = { role: 'assistant', text: '', toolResults: [], interrupt: null };
    setActiveMessage(initialActive);
    activeMessageRef.current = initialActive;

    try {
      const response = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userText,
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('ReadableStream not supported or empty body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent: string | null = null;

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith('event:')) {
            currentEvent = trimmed.slice(6).trim();
          } else if (trimmed.startsWith('data:')) {
            const dataStr = trimmed.slice(5).trim();
            let parsedData: any = {};
            try {
              parsedData = JSON.parse(dataStr);
            } catch (e) {
              parsedData = { text: dataStr };
            }

            handleSSEEvent(currentEvent, parsedData);
            currentEvent = null;
          }
        }
      }
    } catch (err: any) {
      console.error('SSE Stream Error:', err);
      handleSSEEvent('error', { message: err.message || 'Stream connection failed' });
    }
  };

  const handleSSEEvent = (eventType: string | null, data: any) => {
    switch (eventType) {
      case 'token': {
        const tokenText = data.text || '';
        setActiveMessage((prev) => {
          const updated: Message = {
            role: 'assistant',
            toolResults: prev?.toolResults || [],
            interrupt: prev?.interrupt || null,
            text: (prev?.text || '') + tokenText,
          };
          activeMessageRef.current = updated;
          return updated;
        });
        break;
      }

      case 'status': {
        setCurrentStatus(data as StatusData);
        break;
      }

      case 'tool_result': {
        setCurrentStatus(null);
        const { tool, output } = data;

        if (tool === 'generate_quiz_question' || tool === 'grade_answer') {
          setActiveMessage((prev) => {
            const updated: Message = {
              role: 'assistant',
              text: prev?.text || '',
              interrupt: prev?.interrupt || null,
              toolResults: [...(prev?.toolResults || []), { tool, output }],
            };
            activeMessageRef.current = updated;
            return updated;
          });
        }
        break;
      }

      case 'interrupt': {
        setCurrentStatus(null);
        setActiveMessage((prev) => {
          const updated: Message = {
            role: 'assistant',
            ...prev,
            interrupt: data as InterruptData,
          };
          activeMessageRef.current = updated;
          return updated;
        });
        setIsStreaming(false);
        break;
      }

      case 'done': {
        setCurrentStatus(null);
        commitActiveMessage();
        finishTurn();
        break;
      }

      case 'error': {
        setCurrentStatus(null);
        setActiveMessage((prev) => {
          const updated: Message = {
            role: 'assistant',
            ...prev,
            text: (prev?.text || '') + `\n\n*Error: ${data.message || 'An error occurred'}*`,
          };
          activeMessageRef.current = updated;
          return updated;
        });
        commitActiveMessage();
        finishTurn();
        break;
      }

      default:
        break;
    }
  };

  const commitActiveMessage = () => {
    setActiveMessage((currentActive) => {
      if (currentActive) {
        setMessages((prevMessages) => [...prevMessages, currentActive]);
      }
      activeMessageRef.current = null;
      return null;
    });
  };

  const finishTurn = () => {
    setIsStreaming(false);

    if (messageQueue.current.length > 0) {
      const nextUserMessage = messageQueue.current.shift()!;
      executeSendMessage(nextUserMessage);
    }
  };

  const executeSendMessage = (text: string) => {
    setMessages((prev) => [...prev, { role: 'user', text }]);
    sendRequest(text);
  };

  const handleSendMessage = (text: string) => {
    if (isStreaming) {
      messageQueue.current.push(text);
    } else {
      executeSendMessage(text);
    }
  };

  const handleSelectQuizAnswer = (choiceKey: string) => {
    handleSendMessage(`I choose answer option ${choiceKey}`);
  };

  const handleApproveInterrupt = (approved: boolean) => {
    handleSendMessage(approved ? 'Approved' : 'Rejected');
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>LangChain Tutor Agent</h1>
        <span className="thread-badge">Thread: {threadId.slice(0, 8)}...</span>
      </header>

      <ChatWindow
        messages={messages}
        activeMessage={activeMessage}
        onSelectQuizAnswer={handleSelectQuizAnswer}
        onApproveInterrupt={handleApproveInterrupt}
      />

      <StatusIndicator status={currentStatus} />

      <Composer
        onSendMessage={handleSendMessage}
        disabled={isStreaming}
      />
    </div>
  );
}

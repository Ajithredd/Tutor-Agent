import { parse } from 'marked';

interface MessageBubbleProps {
  content: string;
}

export default function MessageBubble({ content }: MessageBubbleProps) {
  const getParsedMarkdown = (text: string) => {
    try {
      const parsed = parse(text || '', { breaks: true });
      return { __html: typeof parsed === 'string' ? parsed : text || '' };
    } catch (e) {
      return { __html: text || '' };
    }
  };

  return (
    <div
      className="markdown-content"
      dangerouslySetInnerHTML={getParsedMarkdown(content)}
    />
  );
}

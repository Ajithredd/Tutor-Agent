export interface QuizQuestionData {
  question: string;
  options: Record<string, string>;
  correct_answer?: string;
  explanation?: string;
}

export interface GradeBadgeData {
  correct?: boolean;
  is_correct?: boolean;
  feedback?: string;
  explanation?: string;
}

export interface ToolResult {
  tool: string;
  output: QuizQuestionData | GradeBadgeData | string | any;
}

export interface StatusData {
  tool: string;
  phase: string;
}

export interface InterruptData {
  action?: string;
  prompt?: string;
  [key: string]: any;
}

export interface Message {
  role: 'user' | 'assistant';
  text?: string;
  toolResults?: ToolResult[];
  interrupt?: InterruptData | null;
}

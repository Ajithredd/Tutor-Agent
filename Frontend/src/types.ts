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

export interface TraceEvent {
  id: string;
  type: 'node_start' | 'node_end' | 'llm_start' | 'tool_start' | 'tool_end';
  node?: string;
  model?: string;
  tool?: string;
  input?: any;
  output?: any;
  timestamp: string;
  run_id?: string;
}

export interface StudentTopicStats {
  correct: number;
  incorrect: number;
  weak_spots: string[];
  last_seen: string;
}

export type StudentProfile = Record<string, StudentTopicStats>;

export interface Message {
  role: 'user' | 'assistant';
  text?: string;
  toolResults?: ToolResult[];
  interrupt?: InterruptData | null;
  traces?: TraceEvent[];
}

import React, { useState } from 'react';
import { TraceEvent, StudentProfile } from '../types';

interface ObservabilityPanelProps {
  traces: TraceEvent[];
  profile: StudentProfile | null;
  isOpen: boolean;
  onToggle: () => void;
  onClearTraces: () => void;
}

export default function ObservabilityPanel({
  traces,
  profile,
  isOpen,
  onToggle,
  onClearTraces,
}: ObservabilityPanelProps) {
  const [activeTab, setActiveTab] = useState<'traces' | 'profile' | 'graph'>('traces');
  const [expandedTraceId, setExpandedTraceId] = useState<string | null>(null);

  const getTraceIcon = (type: string) => {
    switch (type) {
      case 'node_start':
        return '🟢';
      case 'node_end':
        return '🏁';
      case 'llm_start':
        return '🤖';
      case 'tool_start':
        return '⚡';
      case 'tool_end':
        return '📦';
      default:
        return '🔍';
    }
  };

  const getTraceBadgeClass = (type: string) => {
    switch (type) {
      case 'node_start':
      case 'node_end':
        return 'badge-node';
      case 'llm_start':
        return 'badge-llm';
      case 'tool_start':
      case 'tool_end':
        return 'badge-tool';
      default:
        return 'badge-default';
    }
  };

  return (
    <aside className={`observability-panel ${isOpen ? 'open' : 'closed'}`}>
      <div className="panel-header">
        <div className="panel-title-group">
          <span className="live-dot"></span>
          <h2>Observability & Tracing</h2>
        </div>
        <div className="panel-actions">
          <button className="clear-btn" onClick={onClearTraces} title="Clear Trace History">
            Clear
          </button>
          <button className="close-btn" onClick={onToggle} title="Close Panel">
            ✕
          </button>
        </div>
      </div>

      <div className="panel-tabs">
        <button
          className={`panel-tab ${activeTab === 'traces' ? 'active' : ''}`}
          onClick={() => setActiveTab('traces')}
        >
          Traces ({traces.length})
        </button>
        <button
          className={`panel-tab ${activeTab === 'graph' ? 'active' : ''}`}
          onClick={() => setActiveTab('graph')}
        >
          Agent Graph
        </button>
        <button
          className={`panel-tab ${activeTab === 'profile' ? 'active' : ''}`}
          onClick={() => setActiveTab('profile')}
        >
          Student Memory
        </button>
      </div>

      <div className="panel-content">
        {/* TRACES TAB */}
        {activeTab === 'traces' && (
          <div className="traces-timeline">
            {traces.length === 0 ? (
              <div className="empty-state">
                <p>No trace events recorded yet.</p>
                <span>Send a message or answer a quiz to observe live multi-agent execution steps.</span>
              </div>
            ) : (
              traces.map((trace) => {
                const isExpanded = expandedTraceId === trace.id;
                const hasDetails = trace.input || trace.output;

                return (
                  <div
                    key={trace.id}
                    className={`trace-item ${trace.type} ${hasDetails ? 'clickable' : ''}`}
                    onClick={() => hasDetails && setExpandedTraceId(isExpanded ? null : trace.id)}
                  >
                    <div className="trace-item-header">
                      <span className="trace-icon">{getTraceIcon(trace.type)}</span>
                      <span className={`trace-type-badge ${getTraceBadgeClass(trace.type)}`}>
                        {trace.type.replace('_', ' ').toUpperCase()}
                      </span>
                      <span className="trace-title">
                        {trace.tool || trace.node || trace.model || 'Agent Step'}
                      </span>
                      <span className="trace-timestamp">{trace.timestamp}</span>
                    </div>

                    {hasDetails && (
                      <div className="trace-expand-hint">
                        {isExpanded ? 'Hide Payload ▲' : 'View Payload ▼'}
                      </div>
                    )}

                    {isExpanded && (
                      <div className="trace-payload">
                        {trace.input && (
                          <div className="payload-block">
                            <span className="payload-label">Input / Args:</span>
                            <pre>{typeof trace.input === 'object' ? JSON.stringify(trace.input, null, 2) : String(trace.input)}</pre>
                          </div>
                        )}
                        {trace.output && (
                          <div className="payload-block">
                            <span className="payload-label">Output / Result:</span>
                            <pre>{typeof trace.output === 'object' ? JSON.stringify(trace.output, null, 2) : String(trace.output)}</pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* GRAPH ARCHITECTURE TAB */}
        {activeTab === 'graph' && (
          <div className="graph-flow-view">
            <div className="graph-diagram">
              <div className="graph-node-box planner-node">
                <div className="node-badge">Node 1</div>
                <h4>Curriculum Planner</h4>
                <p>Builds 3-step personalized curriculum</p>
              </div>
              <div className="graph-arrow">↓</div>
              <div className="graph-node-box tutor-node">
                <div className="node-badge">Node 2</div>
                <h4>Tutor Explainer (RAG)</h4>
                <p>Retrieves <code>study_material.txt</code> and explains concept</p>
              </div>
              <div className="graph-arrow">↓</div>
              <div className="graph-node-box examiner-node">
                <div className="node-badge">Node 3</div>
                <h4>Examiner & Evaluator</h4>
                <p>Generates quiz, guardrails answer, grades with HITL pause</p>
              </div>
              <div className="graph-arrow">↓ (Adaptive Loop)</div>
              <div className="graph-node-box memory-node">
                <div className="node-badge">State & SQLite</div>
                <h4>AsyncSqliteSaver & Profiles</h4>
                <p>Persists checkpoints, mastery scores & weak spots</p>
              </div>
            </div>
          </div>
        )}

        {/* STUDENT MEMORY TAB */}
        {activeTab === 'profile' && (
          <div className="profile-memory-view">
            {!profile || Object.keys(profile).length === 0 ? (
              <div className="empty-state">
                <p>No student profile stored yet.</p>
                <span>Complete quizzes to track topic mastery and weak spots.</span>
              </div>
            ) : (
              <div className="profile-topic-cards">
                {Object.entries(profile).map(([topic, stats]) => {
                  const total = stats.correct + stats.incorrect;
                  const accuracy = total > 0 ? Math.round((stats.correct / total) * 100) : 0;
                  return (
                    <div key={topic} className="profile-topic-card">
                      <div className="topic-card-header">
                        <h4>{topic}</h4>
                        <span className={`accuracy-pill ${accuracy >= 70 ? 'high' : accuracy >= 40 ? 'medium' : 'low'}`}>
                          {accuracy}% Mastery
                        </span>
                      </div>
                      <div className="topic-stats-row">
                        <span>✅ Correct: <strong>{stats.correct}</strong></span>
                        <span>❌ Incorrect: <strong>{stats.incorrect}</strong></span>
                      </div>
                      {stats.weak_spots && stats.weak_spots.length > 0 && (
                        <div className="weak-spots-list">
                          <strong>Identified Weak Spots:</strong>
                          <ul>
                            {stats.weak_spots.map((ws, i) => (
                              <li key={i}>{ws}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div className="last-seen">Last session: {stats.last_seen || 'Today'}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}

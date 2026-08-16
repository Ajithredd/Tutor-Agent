# 🎓 AI Tutor Agent

An intelligent, interactive, stateful AI Tutoring application powered by **LangChain**, **LangGraph**, **FastAPI**, and **React (Vite)**. The agent adapts to each student's mastery level, generates dynamic quiz questions, grades responses, tracks progress across study sessions, and provides interactive Human-In-The-Loop (HITL) confirmation flows.

---

## ✨ Features

- **🧠 Adaptive Tutoring & Graph Engine**: Built using `LangGraph` and `LangChain`, creating a stateful agent that explains concepts, quizzes students, and dynamically adjusts difficulty based on mastery scores.
- **⚡ Real-Time Streaming UI**: Server-Sent Events (SSE) streaming response protocol connecting a modern React frontend with FastAPI backend.
- **💾 Stateful Checkpointing & Memory**: SQLite persistence (`AsyncSqliteSaver`) preserves conversation context and student mastery state across sessions.
- **👤 Profile & Progress Tracking**: Real-time evaluation of student mastery per topic (correct/incorrect counts and identified weak spots).
- **🙋 Human-In-The-Loop (HITL) Workflows**: Supports node interrupts and interactive student answer confirmations using `LangGraph` commands (`Command(resume=...)`).
- **🛡️ Custom Agent Middlewares**: Custom logging middleware (`GradingLoggingMiddleware`) and fallback handlers to log grading outputs and retry model calls gracefully.

---

## 🛠️ Architecture Overview

```
                 +-----------------------+
                 | React Frontend (Vite) |
                 +-----------+-----------+
                             | (HTTP / SSE)
                             v
                 +-----------------------+
                 |    FastAPI Server     |
                 +-----------+-----------+
                             |
                   +---------v---------+
                   |  LangGraph Agent  |
                   +----+----+----+----+
                        |    |    |
       +----------------+    |    +----------------+
       v                     v                     v
+--------------+    +------------------+    +--------------+
| SQLite DB    |    | Custom Tools     |    | LLM API      |
| Checkpointer |    | (Explain, Quiz,  |    | (Groq/Gemini)|
+--------------+    |  Grade, Mastery) |    +--------------+
                    +------------------+
```

---

## 📂 Project Structure

```
Tutor Agent/
├── Backend/
│   ├── agent.py            # LangGraph agent definitions, state graph, & middlewares
│   ├── server.py           # FastAPI backend server with SSE streaming endpoints
│   ├── tools.py            # Custom tutoring tools & SQLite database functions
│   ├── schemas.py          # Pydantic schemas for data validation
│   ├── study_material.txt  # Core study materials reference file
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Environment variables template
│   └── checkpoints.db      # SQLite database for state checkpointing
├── Frontend/
│   ├── src/                # React source files (components, styles, App.tsx)
│   ├── index.html          # HTML entry point
│   ├── package.json        # Frontend dependencies & scripts
│   ├── vite.config.js      # Vite build configuration
│   └── tsconfig.json       # TypeScript configuration
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 📋 Prerequisites

- **Python**: `3.10` or higher
- **Node.js**: `v18` or higher (`npm` included)
- **API Keys**: Groq API Key (`GROQ_API_KEY`) or Google Gemini API Key (`GOOGLE_API_KEY`)

---

### 🐍 1. Backend Setup

1. **Navigate to the Backend folder:**
   ```bash
   cd Backend
   ```

2. **Create and activate a Virtual Environment:**
   - On **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate
     ```
   - On **macOS/Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file inside the `Backend/` directory (or copy from `.env.example`):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   # OR
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Start the FastAPI Server:**
   ```bash
   uvicorn server:app --reload --port 8000
   ```
   The backend API will run at `http://localhost:8000`.

---

### ⚛️ 2. Frontend Setup

1. **Navigate to the Frontend folder:**
   ```bash
   cd Frontend
   ```

2. **Install Dependencies:**
   ```bash
   npm install
   ```

3. **Start the Development Server:**
   ```bash
   npm run dev
   ```
   The application UI will run at `http://localhost:5173`.

---

## 💻 Usage

1. Open your browser and go to `http://localhost:5173`.
2. Start a conversation with the AI Tutor by greeting it or asking for a specific topic (e.g., *"Explain Python recursion"*).
3. The tutor will explain the concept and prompt you to take a quiz.
4. Select or submit your answer. The agent will grade your response, update your mastery score, and adapt future difficulty level accordingly!

---

## 🔧 Technologies Used

- **Backend**: Python, FastAPI, LangChain, LangGraph, SQLite, Uvicorn, SSE-Starlette
- **Frontend**: React 18, TypeScript, Vite, Marked (Markdown rendering), CSS3
- **LLM Integrations**: Groq (`llama-3.3-70b-versatile`), Google Gemini (`gemini-1.5-flash`)

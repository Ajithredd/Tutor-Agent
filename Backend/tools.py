import os
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.tools import tool
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnableConfig
# pyrefly: ignore [missing-import]
from langchain_core.vectorstores import InMemoryVectorStore
# pyrefly: ignore [missing-import]
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# Commented-out alternative model:
# from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import QuizQuestion, GradeResult

load_dotenv()

# Persistent SQLite database connection
DB_PATH = "checkpoints.db"
db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# Setup schemas
def setup_databases():
    # Setup student profiles schema
    cursor = db_conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profiles (
            thread_id TEXT,
            topic TEXT,
            correct INTEGER DEFAULT 0,
            incorrect INTEGER DEFAULT 0,
            weak_spots TEXT DEFAULT '',
            last_seen TEXT DEFAULT '',
            PRIMARY KEY (thread_id, topic)
        )
    """)
    db_conn.commit()

setup_databases()

# Setup Vector Store for RAG
vector_store = InMemoryVectorStore(GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2"))

def ingest_study_material():
    material_path = "study_material.txt"
    if not os.path.exists(material_path):
        return
    with open(material_path, "r", encoding="utf-8") as f:
        content = f.read()
    sections = content.split("===")
    documents = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        paragraphs = section.split("\n\n")
        header = paragraphs[0] if paragraphs else "General Study Material"
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            text = f"{header}\n{p}" if p != header else p
            documents.append(Document(page_content=text, metadata={"source": "study_material.txt"}))
    vector_store.add_documents(documents)

try:
    ingest_study_material()
except Exception as e:
    print(f"Error ingesting study material: {e}")

# Hardcoded model initialization using ChatGroq
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

# Commented-out alternative model declaration for fast swap:
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

@tool
def explain_concept(topic: str, student_level: str = "beginner") -> str:
    """Explains a concept to a student clearly and concisely based on their level, grounding it in reference materials if available."""
    passages = ""
    try:
        results = vector_store.similarity_search(topic, k=2)
        if results:
            passages = "\n".join([doc.page_content for doc in results])
    except Exception:
        pass
        
    if passages:
        prompt = (
            f"Explain the concept of '{topic}' for a {student_level} student concisely.\n"
            f"You MUST base your explanation on the following reference material:\n"
            f"\"\"\"\n{passages}\n\"\"\"\n\n"
            f"Provide a clear, simple explanation. At the very end of your response, "
            f"you MUST append a source reference block containing a snippet of the matching source notes, "
            f"formatted exactly as: '[Source: study_material.txt - \"snippet of notes\"]'.\n"
            f"Do not ask any trailing questions."
        )
    else:
        prompt = f"Explain the concept of '{topic}' for a {student_level} student concisely. Do not ask any trailing questions."
        
    response = model.invoke(prompt)
    return str(response.content)

@tool
def generate_quiz_question(topic: str, difficulty: str = "medium") -> dict:
    """Generates a high quality multiple-choice quiz question (options A, B, C, D) on a given topic and difficulty."""
    structured_llm = model.with_structured_output(QuizQuestion)
    prompt = (
        f"You are creating a multiple-choice quiz question on the topic: '{topic}' with difficulty '{difficulty}'.\n"
        f"CRITICAL REQUIREMENTS:\n"
        f"1. Provide exactly four realistic, distinct options labeled 'A', 'B', 'C', and 'D'.\n"
        f"2. Each option value MUST be actual answer text (e.g. for 'What is the capital of France?', options must be {{\"A\": \"Paris\", \"B\": \"Berlin\", \"C\": \"Rome\", \"D\": \"Madrid\"}}).\n"
        f"3. Do NOT make the options repetitive placeholders (e.g., do NOT write 'The capital of France', 'The capital of Germany').\n"
        f"4. Set 'correct_answer' strictly to the option key corresponding to the correct answer (e.g., 'A', 'B', 'C', or 'D').\n"
        f"5. Provide a clear explanation of why that option is correct."
    )
    result = structured_llm.invoke(prompt)
    if isinstance(result, QuizQuestion):
        return result.model_dump()
    return dict(result)

@tool
def grade_answer(question: str, correct_answer: str, student_answer: str) -> dict:
    """Grades a student's answer against the specific question and correct answer key/text, providing clear feedback."""
    structured_llm = model.with_structured_output(GradeResult)
    prompt = (
        f"You are an accurate, encouraging quiz evaluator.\n"
        f"QUESTION: {question}\n"
        f"EXPECTED CORRECT ANSWER: {correct_answer}\n"
        f"STUDENT'S SUBMISSION: {student_answer}\n\n"
        f"GRADING RULES:\n"
        f"1. Identify the option letter or text the student chose (e.g., if student says 'I choose answer option A' or 'A' or 'Paris', their chosen answer is 'A' / 'Paris').\n"
        f"2. Check if their chosen option matches the EXPECTED CORRECT ANSWER ({correct_answer}).\n"
        f"3. If they match: set 'correct' to true, set 'feedback' praising them concisely, and set 'correct_answer' to '{correct_answer}'.\n"
        f"4. If they do NOT match: set 'correct' to false, set 'feedback' explaining what their choice was and why '{correct_answer}' is the correct answer.\n"
        f"5. DO NOT invent or swap the student's answer (if the student selected A, do not say they selected B)."
    )
    result = structured_llm.invoke(prompt)
    if isinstance(result, GradeResult):
        return result.model_dump()
    return dict(result)

# Database Helper functions
def db_get_student_profile(thread_id: str) -> dict:
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT topic, correct, incorrect, weak_spots, last_seen
        FROM student_profiles
        WHERE thread_id = ?
    """, (thread_id,))
    rows = cursor.fetchall()
    profile = {}
    for row in rows:
        topic, correct, incorrect, weak_spots, last_seen = row
        profile[topic] = {
            "correct": correct,
            "incorrect": incorrect,
            "weak_spots": [s.strip() for s in weak_spots.split(",") if s.strip()] if weak_spots else [],
            "last_seen": last_seen
        }
    return profile

def db_update_student_profile(thread_id: str, topic: str, is_correct: bool, weak_spot: str | None = None):
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT correct, incorrect, weak_spots FROM student_profiles
        WHERE thread_id = ? AND topic = ?
    """, (thread_id, topic))
    row = cursor.fetchone()
    
    today_str = datetime.now().date().isoformat()
    
    if row:
        correct, incorrect, weak_spots_str = row
        weak_spots = [s.strip() for s in weak_spots_str.split(",") if s.strip()] if weak_spots_str else []
        if is_correct:
            correct += 1
        else:
            incorrect += 1
            if weak_spot and weak_spot not in weak_spots:
                weak_spots.append(weak_spot)
        new_weak_spots_str = ",".join(weak_spots)
        cursor.execute("""
            UPDATE student_profiles
            SET correct = ?, incorrect = ?, weak_spots = ?, last_seen = ?
            WHERE thread_id = ? AND topic = ?
        """, (correct, incorrect, new_weak_spots_str, today_str, thread_id, topic))
    else:
        correct = 1 if is_correct else 0
        incorrect = 0 if is_correct else 1
        new_weak_spots_str = weak_spot if weak_spot else ""
        cursor.execute("""
            INSERT INTO student_profiles (thread_id, topic, correct, incorrect, weak_spots, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (thread_id, topic, correct, incorrect, new_weak_spots_str, today_str))
    
    db_conn.commit()

@tool
def get_student_profile(config: RunnableConfig) -> dict:
    """Retrieves the persistent student profile (mastery scores, weak spots, last seen date) for the current thread/session."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    return db_get_student_profile(thread_id)

@tool
def update_mastery_score(topic: str, is_correct: bool, config: RunnableConfig) -> str:
    """Updates the student's mastery score (correct/incorrect count) for a specific topic in the current session.
    
    Args:
        topic: The learning topic/subject (e.g., 'Python variables', 'Photosynthesis').
        is_correct: True if the student's answer was correct, False otherwise.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    db_update_student_profile(thread_id, topic, is_correct)
    profile = db_get_student_profile(thread_id).get(topic, {})
    return f"Mastery score for '{topic}' updated: {profile.get('correct')} correct, {profile.get('incorrect')} incorrect."

@tool
def get_mastery_scores(config: RunnableConfig) -> dict:
    """Retrieves all mastery scores for the current session. Use this to check the student's performance history."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    profile = db_get_student_profile(thread_id)
    return {topic: {"correct": data["correct"], "incorrect": data["incorrect"]} for topic, data in profile.items()}

TUTOR_TOOLS = [
    explain_concept,
    generate_quiz_question,
    grade_answer,
    update_mastery_score,
    get_mastery_scores,
    get_student_profile,
]

@tool
def retrieve_reference(query: str) -> str:
    """Retrieves relevant study notes and concepts from the reference study material to ground explanations.
    
    Args:
        query: Search query string (e.g. 'Calvin cycle stroma', 'break vs continue').
    """
    try:
        results = vector_store.similarity_search(query, k=2)
        if not results:
            return "No matching reference materials found."
        formatted = "Reference Material Passages:\n"
        for idx, doc in enumerate(results):
            formatted += f"[{idx+1}] {doc.page_content}\n"
        return formatted
    except Exception as e:
        return f"Error retrieving reference material: {str(e)}"

TUTOR_TOOLS.append(retrieve_reference)

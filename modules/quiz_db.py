# modules/quiz_db.py
import sqlite3
import json
import os
from datetime import datetime

def init_quiz_db(db_file: str):
    """Initializes the quiz database with required tables."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            num_questions INTEGER,
            difficulty TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question_number INTEGER,
            question_text TEXT NOT NULL,
            options TEXT NOT NULL,
            correct_answer INTEGER NOT NULL,
            explanation TEXT,
            source_file TEXT,
            FOREIGN KEY (session_id) REFERENCES quiz_sessions (id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            user_answer INTEGER,
            is_correct INTEGER,
            FOREIGN KEY (session_id) REFERENCES quiz_sessions (id),
            FOREIGN KEY (question_id) REFERENCES quiz_questions (id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL UNIQUE,
            score INTEGER,
            total_questions INTEGER,
            percentage REAL,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES quiz_sessions (id)
        )
    """)
    
    conn.commit()
    conn.close()


def create_quiz_session(db_file: str, model_name: str, num_questions: int, difficulty: str):
    """Creates a new quiz session."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO quiz_sessions (model_name, num_questions, difficulty) VALUES (?, ?, ?)",
        (model_name, num_questions, difficulty)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def add_quiz_questions(db_file: str, session_id: int, quiz_data: dict):
    """Adds generated quiz questions to a session.
    
    quiz_data should be in format:
    {
        "quiz": [
            {
                "question": "...",
                "options": [...],
                "correct_answer": 0,
                "explanation": "..."
            }
        ]
    }
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    for i, q in enumerate(quiz_data.get("quiz", []), start=1):
        cursor.execute(
            """INSERT INTO quiz_questions 
               (session_id, question_number, question_text, options, correct_answer, explanation)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                i,
                q.get("question", ""),
                json.dumps(q.get("options", [])),
                q.get("correct_answer", 0),
                q.get("explanation", "")
            )
        )
    
    conn.commit()
    conn.close()


def add_quiz_response(db_file: str, session_id: int, question_id: int, user_answer: int, is_correct: bool):
    """Records a user's answer to a quiz question."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO quiz_responses 
           (session_id, question_id, user_answer, is_correct)
           VALUES (?, ?, ?, ?)""",
        (session_id, question_id, user_answer, int(is_correct))
    )
    conn.commit()
    conn.close()


def add_quiz_score(db_file: str, session_id: int, score: int, total: int):
    """Records the final score for a quiz session."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    percentage = (score / total * 100) if total > 0 else 0
    
    cursor.execute(
        """INSERT INTO quiz_scores 
           (session_id, score, total_questions, percentage)
           VALUES (?, ?, ?, ?)""",
        (session_id, score, total, percentage)
    )
    conn.commit()
    conn.close()


def get_quiz_session(db_file: str, session_id: int):
    """Retrieves quiz session details."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, model_name, num_questions, difficulty, created_at FROM quiz_sessions WHERE id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "model_name": row[1],
            "num_questions": row[2],
            "difficulty": row[3],
            "created_at": row[4]
        }
    return None


def get_quiz_questions(db_file: str, session_id: int):
    """Retrieves all questions for a quiz session."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, question_number, question_text, options, correct_answer, explanation 
           FROM quiz_questions 
           WHERE session_id = ? 
           ORDER BY question_number ASC""",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    questions = []
    for row in rows:
        questions.append({
            "id": row[0],
            "question_number": row[1],
            "question_text": row[2],
            "options": json.loads(row[3]),
            "correct_answer": row[4],
            "explanation": row[5]
        })
    return questions


def get_quiz_responses(db_file: str, session_id: int):
    """Retrieves all responses for a quiz session."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT question_id, user_answer, is_correct 
           FROM quiz_responses 
           WHERE session_id = ? 
           ORDER BY question_id ASC""",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    responses = {}
    for row in rows:
        responses[row[0]] = {
            "user_answer": row[1],
            "is_correct": bool(row[2])
        }
    return responses


def get_quiz_score(db_file: str, session_id: int):
    """Retrieves the score for a completed quiz session."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT score, total_questions, percentage, completed_at FROM quiz_scores WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "score": row[0],
            "total": row[1],
            "percentage": row[2],
            "completed_at": row[3]
        }
    return None


def get_all_quiz_sessions(db_file: str, limit: int = 50):
    """Retrieves all quiz sessions, ordered by most recent first."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT s.id, s.model_name, s.num_questions, s.difficulty, s.created_at,
                  sc.score, sc.total_questions, sc.percentage
           FROM quiz_sessions s
           LEFT JOIN quiz_scores sc ON s.id = sc.session_id
           ORDER BY s.created_at DESC
           LIMIT ?""",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    sessions = []
    for row in rows:
        sessions.append({
            "id": row[0],
            "model_name": row[1],
            "num_questions": row[2],
            "difficulty": row[3],
            "created_at": row[4],
            "score": row[5],
            "total": row[6],
            "percentage": row[7]
        })
    return sessions


def delete_quiz_session(db_file: str, session_id: int):
    """Deletes a quiz session and all associated data."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Delete related records
    cursor.execute("DELETE FROM quiz_responses WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM quiz_questions WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM quiz_scores WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM quiz_sessions WHERE id = ?", (session_id,))
    
    conn.commit()
    conn.close()


def get_quiz_stats(db_file: str):
    """Retrieves aggregate statistics about all quizzes."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT 
           COUNT(DISTINCT s.id) as total_quizzes,
           AVG(sc.percentage) as avg_percentage,
           MAX(sc.percentage) as best_percentage,
           MIN(sc.percentage) as worst_percentage
           FROM quiz_sessions s
           LEFT JOIN quiz_scores sc ON s.id = sc.session_id"""
    )
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0] > 0:
        return {
            "total_quizzes": row[0],
            "avg_percentage": row[1] or 0,
            "best_percentage": row[2] or 0,
            "worst_percentage": row[3] or 0
        }
    return {
        "total_quizzes": 0,
        "avg_percentage": 0,
        "best_percentage": 0,
        "worst_percentage": 0
    }
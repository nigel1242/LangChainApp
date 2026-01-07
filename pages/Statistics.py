# pages/03_📊_Statistics.py

from __future__ import annotations
from datetime import datetime
import pandas as pd
import streamlit as st

# --- Local Module Imports ---
from modules.db_manager import DBManager
from modules.quizstats.subject_store import list_subjects
from modules.quizstats.progress_store import (
    init_stats_db,
    list_subjects_with_stats,
    get_attempts_for_subject,
    clear_subject_stats,
)

# ------------------ CONFIGURATION ------------------
st.set_page_config(page_title="Progress Statistics", page_icon="📊", layout="wide")
QUESTIONS_PER_ATTEMPT = 10  # 1 attempt is defined as 10 questions

def init_page():
    """Initializes databases and backend managers."""
    init_stats_db()
    
    # Initialize DBManager to match app.py (handles SQLite or Qdrant)
    return DBManager(
        backend=st.session_state.get("chat_backend", "sqlite"),
        db_file="chat_playground.db",
        qdrant_url=st.session_state.get("qdrant_url"),
        qdrant_api_key=st.session_state.get("qdrant_api_key"),
    )

db = init_page()

# ------------------ SUBJECT SELECTION ------------------
st.title("📊 Progress & Statistics")

# Combine subjects from the Library and those with existing stats
library_subjects = db.load_all_subjects()
stats_subjects = list_subjects_with_stats()
combined_options = sorted(list(set(library_subjects + stats_subjects)))

if not combined_options:
    st.info("No quiz attempts or subjects found. Go take a quiz first!")
    st.stop()

# Use shared session state for subject selection consistency
current_default = st.session_state.get("current_subject", combined_options[0])

selected_subject = st.selectbox(
    "📚 Select Subject to Analyze",
    combined_options,
    index=combined_options.index(current_default) if current_default in combined_options else 0
)

# Sync subject across pages
if selected_subject != st.session_state.get("current_subject"):
    st.session_state.current_subject = selected_subject
    st.rerun()

# ------------------ DATA RETRIEVAL ------------------
attempts = get_attempts_for_subject(selected_subject)

if not attempts:
    st.info(f"No attempts recorded for **{selected_subject}** yet.")
    st.stop()

# ------------------ RESET UTILITY ------------------
with st.expander("⚠️ Danger Zone"):
    if st.button(f"🗑️ Reset all data for {selected_subject}"):
        clear_subject_stats(selected_subject)
        st.success(f"Cleared all statistics for {selected_subject}.")
        st.rerun()

# ------------------ DATA PROCESSING ------------------
df = pd.DataFrame(attempts)

# Ensure all required columns exist defensively
cols = ["ts_ms", "is_correct", "topic", "question", "chosen", "correct", "difficulty", "source_hint"]
for col in cols:
    if col not in df.columns:
        df[col] = None

# Convert timestamps and filter out invalid/empty rows
df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce").fillna(0).astype(int)
df = df[
    (df["ts_ms"] > 0) & 
    (df["question"].astype(str).str.strip().ne(""))
].sort_values("ts_ms").reset_index(drop=True)

if df.empty:
    st.warning("No valid data points found.")
    st.stop()

# Enrich DataFrame
df["datetime"] = df["ts_ms"].apply(lambda ms: datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M"))
df["correct_bool"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int) == 1

# Handle Topic Fallbacks
def get_clean_topic(row):
    t = str(row.get("topic") or "").strip()
    if t and t.lower() != "none": return t
    diff = str(row.get("difficulty") or "").strip()
    return f"General ({diff})" if diff else "General"

df["topic"] = df.apply(get_clean_topic, axis=1)

# Grouping logic for "Attempts"
df["q_index"] = df.index + 1
df["attempt_id"] = ((df["q_index"] - 1) // QUESTIONS_PER_ATTEMPT) + 1
df["q_in_attempt"] = ((df["q_index"] - 1) % QUESTIONS_PER_ATTEMPT) + 1

# ------------------ VISIBILITY CONTROLS ------------------
max_attempt = int(df["attempt_id"].max())
attempt_counts = df.groupby("attempt_id")["question"].count()
complete_ids = attempt_counts[attempt_counts >= QUESTIONS_PER_ATTEMPT].index.tolist()

ctrl1, ctrl2 = st.columns([1, 2])
with ctrl1:
    show_inc = st.toggle("Include partial attempts", value=False)
    available_ids = list(range(1, max_attempt + 1)) if show_inc else complete_ids
    
    if not available_ids:
        st.warning("No complete attempts (10 questions) found yet.")
        st.stop()

    chosen_id = st.selectbox("View Attempt #", options=available_ids, index=len(available_ids)-1)

# ------------------ METRICS DASHBOARD ------------------
# Scope data based on visibility toggle
df_scope = df if show_inc else df[df["attempt_id"].isin(complete_ids)]
att_df = df[df["attempt_id"] == chosen_id]

# Calculate stats
total_q, total_correct = len(df_scope), int(df_scope["correct_bool"].sum())
att_q, att_correct = len(att_df), int(att_df["correct_bool"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Questions", total_q)
m2.metric("Overall Accuracy", f"{(total_correct/total_q*100):.1f}%" if total_q else "0%")
m3.metric(f"Attempt {chosen_id} Score", f"{att_correct} / {att_q}")
m4.metric(f"Attempt {chosen_id} Accuracy", f"{(att_correct/att_q*100):.1f}%" if att_q else "0%")

st.divider()

# ------------------ TOPIC ANALYSIS ------------------
st.subheader(f"🏷️ Topic Breakdown: Attempt {chosen_id}")
topic_stats = att_df.groupby("topic").agg(
    Questions=("question", "count"),
    Correct=("correct_bool", "sum")
).reset_index()
topic_stats["Accuracy %"] = (topic_stats["Correct"] / topic_stats["Questions"] * 100).round(1)

st.dataframe(topic_stats.sort_values("Accuracy %"), use_container_width=True, hide_index=True)

# ------------------ ATTEMPT HISTORY ------------------
st.subheader("🧾 Score History")
history = df_scope.groupby("attempt_id").agg(
    Start=("datetime", "min"),
    Questions=("question", "count"),
    Correct=("correct_bool", "sum")
).reset_index()

history["Score"] = history.apply(lambda r: f"{int(r['Correct'])} / {int(r['Questions'])}", axis=1)
history["Accuracy %"] = (history["Correct"] / history["Questions"] * 100).round(1)

st.dataframe(
    history.sort_values("attempt_id", ascending=False), 
    use_container_width=True, 
    hide_index=True
)

# ------------------ MISTAKE TRACKER ------------------
st.subheader("❌ Recent Mistakes")
mistakes = df_scope[df_scope["correct_bool"] == False].sort_values("ts_ms", ascending=False).head(30)

if mistakes.empty:
    st.success("No mistakes found! Keep it up.")
else:
    view_cols = ["datetime", "attempt_id", "topic", "question", "chosen", "correct", "source_hint"]
    st.dataframe(
        mistakes[view_cols].rename(columns={
            "datetime": "Time", "attempt_id": "Att#", "chosen": "Your Answer", "correct": "Correct Answer"
        }), 
        use_container_width=True, 
        hide_index=True
    )
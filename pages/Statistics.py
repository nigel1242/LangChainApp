# pages/03_📊_Statistics.py

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from modules.quizstats.subject_store import list_subjects
from modules.quizstats.progress_store import (
    init_stats_db,
    list_subjects_with_stats,
    get_attempts_for_subject,
    clear_subject_stats,
)

st.set_page_config(page_title="Progress Statistics", page_icon="📊", layout="wide")
st.title("📊 Progress & Statistics")

# ---------------- CONFIG ----------------
QUESTIONS_PER_ATTEMPT = 10  # ✅ 1 attempt = 10 questions

init_stats_db()

# ---------- subject selector ----------
all_subjects_defined = [s["name"] for s in list_subjects()]
all_subjects_with_stats = list_subjects_with_stats()

if not all_subjects_with_stats:
    st.info("No quiz attempts recorded yet. Do some quizzes first!")
    st.stop()

options = sorted(set(all_subjects_with_stats + all_subjects_defined))
subject = st.selectbox("Choose subject", options)

attempts = get_attempts_for_subject(subject)
if not attempts:
    st.info("No attempts recorded for this subject yet.")
    st.stop()

# ---------- reset stats ----------
st.markdown("---")
st.subheader("⚠️ Reset Statistics")
if st.button("🗑️ Clear ALL recorded data for this subject"):
    clear_subject_stats(subject)
    st.success(f"All statistics for **{subject}** have been cleared.")
    st.rerun()

st.markdown("---")

# ---------- Load dataframe ----------
df = pd.DataFrame(attempts)

# normalize columns defensively
for col in ["ts_ms", "is_correct", "topic", "question", "chosen", "correct", "difficulty", "source_hint"]:
    if col not in df.columns:
        df[col] = None

df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce").fillna(0).astype(int)
df = df.sort_values("ts_ms").reset_index(drop=True)

# ===============================
# 🚫 DROP MEANINGLESS / INVALID ROWS (fixes the “empty first row” issue)
# - remove rows with ts_ms == 0
# - remove rows with blank question/correct
# ===============================
df = df[
    (df["ts_ms"] > 0)
    & df["question"].astype(str).str.strip().ne("")
    & df["correct"].astype(str).str.strip().ne("")
].copy()

# re-sort after filtering
df = df.sort_values("ts_ms").reset_index(drop=True)

# if everything got filtered out
if df.empty:
    st.info("No valid attempts found yet (only empty/invalid rows were recorded).")
    st.stop()

df["datetime"] = df["ts_ms"].apply(
    lambda ms: datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
)
df["correct_bool"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int) == 1

# topic cleanup (avoid blanks; also avoid “(No topic)” by falling back to something meaningful)
df["topic"] = df["topic"].fillna("").astype(str).str.strip()

# If topic is blank, fall back to something consistent across subjects:
# 1) difficulty (if exists)
# 2) file name from source_hint (if it contains something)
# 3) "General"
def _fallback_topic(row) -> str:
    t = (row.get("topic") or "").strip()
    if t:
        return t

    diff = (row.get("difficulty") or "").strip()
    if diff:
        return f"General ({diff})"

    src = (row.get("source_hint") or "").strip()
    if src:
        return "General (from notes)"

    return "General"

df["topic"] = df.apply(_fallback_topic, axis=1)

# ---------- Build "attempt_id" (1 attempt = 10 questions) ----------
df["q_index_in_subject"] = df.index + 1
df["attempt_id"] = ((df["q_index_in_subject"] - 1) // QUESTIONS_PER_ATTEMPT) + 1
df["q_in_attempt"] = ((df["q_index_in_subject"] - 1) % QUESTIONS_PER_ATTEMPT) + 1

# ---------- Option: only show complete attempts ----------
max_attempt = int(df["attempt_id"].max())
attempt_counts = df.groupby("attempt_id")["question"].count()
complete_attempt_ids = attempt_counts[attempt_counts >= QUESTIONS_PER_ATTEMPT].index.tolist()

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    show_incomplete = st.toggle("Include incomplete attempts", value=False)
with colB:
    if show_incomplete:
        available_attempts = list(range(1, max_attempt + 1))
    else:
        available_attempts = complete_attempt_ids

    if not available_attempts:
        st.info("No complete attempts yet (need 10 answered questions to form 1 attempt).")
        st.stop()

    chosen_attempt = st.selectbox(
        "View attempt",
        options=available_attempts,
        index=len(available_attempts) - 1,  # default to latest
    )

with colC:
    st.caption(
        f"Attempts are computed as **every {QUESTIONS_PER_ATTEMPT} questions** (time-ordered) for this subject."
    )

# ---------- Attempt-level summary ----------
attempt_df = df[df["attempt_id"] == chosen_attempt].copy()
attempt_total_q = len(attempt_df)
attempt_correct = int(attempt_df["correct_bool"].sum())
attempt_acc = (attempt_correct / attempt_total_q * 100) if attempt_total_q else 0.0

# overall (subject) but based on chosen visibility
if show_incomplete:
    df_scope = df.copy()
else:
    df_scope = df[df["attempt_id"].isin(complete_attempt_ids)].copy()

total_q = len(df_scope)
total_correct = int(df_scope["correct_bool"].sum())
overall_acc = (total_correct / total_q * 100) if total_q else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Subject questions recorded", total_q)
m2.metric("Overall accuracy", f"{overall_acc:.1f}%")
m3.metric(f"Attempt {chosen_attempt} score", f"{attempt_correct} / {attempt_total_q}")
m4.metric(f"Attempt {chosen_attempt} accuracy", f"{attempt_acc:.1f}%")

st.markdown("---")

# =========================
# ✅ ATTEMPT: TOPIC BREAKDOWN
# =========================
st.subheader(f"🏷️ Attempt {chosen_attempt}: Topic scores (relative to topic)")

topic_attempt = (
    attempt_df.groupby("topic", as_index=False)
    .agg(
        questions=("question", "count"),
        correct=("correct_bool", "sum"),
    )
)
topic_attempt["accuracy_pct"] = (topic_attempt["correct"] / topic_attempt["questions"] * 100).round(1)
topic_attempt = topic_attempt.sort_values(["accuracy_pct", "questions"], ascending=[True, False])

st.dataframe(
    topic_attempt.rename(
        columns={
            "topic": "Topic",
            "questions": "Questions",
            "correct": "Correct",
            "accuracy_pct": "Accuracy (%)",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# =========================
# ✅ ATTEMPT HISTORY TABLE (score out of 10)
# =========================
st.markdown("---")
st.subheader("🧾 Attempt history (score per attempt)")

attempt_history = (
    df_scope.groupby("attempt_id", as_index=False)
    .agg(
        start_time=("ts_ms", "min"),
        end_time=("ts_ms", "max"),
        questions=("question", "count"),
        correct=("correct_bool", "sum"),
    )
)

attempt_history["start_time"] = attempt_history["start_time"].apply(
    lambda ms: datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
)
attempt_history["end_time"] = attempt_history["end_time"].apply(
    lambda ms: datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
)
attempt_history["score"] = attempt_history.apply(
    lambda r: f"{int(r['correct'])} / {int(r['questions'])}", axis=1
)
attempt_history["accuracy_pct"] = (attempt_history["correct"] / attempt_history["questions"] * 100).round(1)

attempt_history = attempt_history.sort_values("attempt_id", ascending=False)

st.dataframe(
    attempt_history.rename(
        columns={
            "attempt_id": "Attempt",
            "start_time": "Start",
            "end_time": "End",
            "questions": "Questions",
            "correct": "Correct",
            "score": "Score",
            "accuracy_pct": "Accuracy (%)",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# =========================
# ✅ Recent mistakes (filtered to remove empty/meaningless rows)
# =========================
st.markdown("---")
st.subheader("❌ Recent mistakes")

wrong_df = df_scope[df_scope["correct_bool"] == False].copy()

# remove rows that have no useful content (prevents blank rows in this table too)
wrong_df = wrong_df[
    wrong_df["question"].astype(str).str.strip().ne("")
    & wrong_df["correct"].astype(str).str.strip().ne("")
].copy()

if wrong_df.empty:
    st.success("Nice! No wrong answers recorded yet.")
else:
    wrong_df = wrong_df.sort_values("ts_ms", ascending=False).head(30)
    view = wrong_df[
        ["datetime", "attempt_id", "q_in_attempt", "topic", "question", "chosen", "correct", "difficulty", "source_hint"]
    ].rename(
        columns={
            "datetime": "When",
            "attempt_id": "Attempt",
            "q_in_attempt": "Q#",
            "topic": "Topic",
            "question": "Question",
            "chosen": "Your answer",
            "correct": "Correct answer",
            "difficulty": "Difficulty",
            "source_hint": "Source snippet",
        }
        
    )
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("These are your 30 most recent incorrect answers.")

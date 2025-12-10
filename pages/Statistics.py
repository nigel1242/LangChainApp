# pages/03_📊_Statistics.py

from __future__ import annotations

import math
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

# --- CLEAR DATA BUTTON ---
st.markdown("---")
st.subheader("⚠️ Reset Statistics")

if st.button("🗑️ Clear ALL recorded data for this subject"):
    clear_subject_stats(subject)
    st.success(f"All statistics for **{subject}** have been cleared.")
    st.rerun()

st.markdown("---")

# ---------- Load dataframe ----------
df = pd.DataFrame(attempts)
df["datetime"] = df["ts_ms"].apply(
    lambda ms: datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M")
)
df["correct_bool"] = df["is_correct"] == 1

# ---------- summary metrics ----------
total = len(df)
correct = int(df["correct_bool"].sum())
accuracy = correct / total * 100 if total else 0

# last 20 attempts accuracy
recent_n = min(20, total)
recent_df = df.tail(recent_n)
recent_acc = (
    recent_df["correct_bool"].sum() / recent_n * 100 if recent_n > 0 else math.nan
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total attempts", total)
with col2:
    st.metric("Overall accuracy", f"{accuracy:.1f}%")
with col3:
    st.metric(f"Accuracy (last {recent_n})", f"{recent_acc:.1f}%")

st.markdown("---")

# ---------- accuracy over time ----------
st.subheader("📈 Accuracy over time")

df_sorted = df.sort_values("ts_ms").reset_index(drop=True)
df_sorted["attempt"] = df_sorted.index + 1
df_sorted["cumulative_correct"] = df_sorted["correct_bool"].cumsum()
df_sorted["cumulative_accuracy"] = (
    df_sorted["cumulative_correct"] / df_sorted["attempt"] * 100
)

line_df = df_sorted[["attempt", "cumulative_accuracy"]].set_index("attempt")
st.line_chart(line_df)

st.caption("Each point shows your cumulative accuracy up to that attempt.")

st.markdown("---")

# ---------- recent mistakes ----------
st.subheader("❌ Recent mistakes")

wrong_df = df[df["correct_bool"] == False].copy()
if wrong_df.empty:
    st.success("Nice! No wrong answers recorded yet.")
else:
    wrong_df = wrong_df.sort_values("ts_ms", ascending=False).head(30)
    wrong_df_view = wrong_df[
        ["datetime", "question", "chosen", "correct", "topic", "source_hint"]
    ]
    wrong_df_view = wrong_df_view.rename(
        columns={
            "datetime": "When",
            "question": "Question",
            "chosen": "Your answer",
            "correct": "Correct answer",
            "topic": "Topic / Image hint",
            "source_hint": "Source snippet",
        }
    )
    st.dataframe(wrong_df_view, use_container_width=True)
    st.caption("These are your 30 most recent incorrect answers.")

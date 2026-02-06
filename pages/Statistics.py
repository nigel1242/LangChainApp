from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher
import os
import re

import pandas as pd
import streamlit as st

# --- Local Module Imports ---
from modules.db_manager import DBManager
from modules.quizstats.progress_store import (
    init_stats_db,
    get_attempts_for_subject,
    clear_subject_stats,
)

# ------------------ CONFIGURATION ------------------
st.set_page_config(page_title="Progress Statistics", page_icon="📊", layout="wide")
QUESTIONS_PER_ATTEMPT = 10
CHAT_DB_FILE = "chat.db"


# ============================================================
#   TOPIC GROUPING (NO HARD-CODED REPLACEMENTS)
# ============================================================
def _topic_clean_base(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\.(pdf|pptx|docx|txt|csv)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*[-–—]\s*", " – ", t).strip()
    if not t:
        return "General"
    if len(t) > 80:
        t = t[:80].rstrip() + "..."
    return t


def build_topic_alias_map(topics: list[str], threshold: float = 0.88) -> dict[str, str]:
    """
    Cluster similar topics using SequenceMatcher ratio.
    Returns mapping: original_topic -> representative_topic
    """
    if not topics:
        return {}

    cleaned = [_topic_clean_base(t) for t in topics]
    counts: dict[str, int] = {}
    originals_by_clean: dict[str, list[str]] = {}

    for orig, c in zip(topics, cleaned):
        counts[c] = counts.get(c, 0) + 1
        originals_by_clean.setdefault(c, []).append(orig)

    candidates = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)

    clusters: list[list[str]] = []
    used = set()

    for anchor in candidates:
        if anchor in used:
            continue
        cluster = [anchor]
        used.add(anchor)

        for other in candidates:
            if other in used:
                continue
            ratio = SequenceMatcher(None, anchor.lower(), other.lower()).ratio()
            if ratio >= threshold:
                cluster.append(other)
                used.add(other)

        clusters.append(cluster)

    rep_map: dict[str, str] = {}
    for cluster in clusters:
        # pick most frequent, then shortest
        cluster_sorted = sorted(cluster, key=lambda x: (-counts.get(x, 0), len(x)))
        rep_clean = cluster_sorted[0]

        origs = originals_by_clean.get(rep_clean, [rep_clean])
        rep_orig = sorted((_topic_clean_base(o) for o in origs), key=len)[0]
        rep = rep_orig or rep_clean

        for c in cluster:
            rep_map[c] = rep

    alias: dict[str, str] = {}
    for orig, c in zip(topics, cleaned):
        alias[orig] = rep_map.get(c, _topic_clean_base(orig))

    return alias


# ============================================================
#   MATH RENDERING FIXES (similar to quiz_engine.py)
# ============================================================
_LATEX_TRIGGERS = (
    r"\frac",
    r"\dfrac",
    r"\tfrac",
    r"\left",
    r"\right",
    r"\sin",
    r"\cos",
    r"\tan",
    r"\ln",
    r"\log",
    r"\sqrt",
    r"\int",
    r"\sum",
    r"\lim",
    r"\mathcal",
    r"\cdot",
    r"\times",
    r"\pm",
)


def _looks_like_math(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False

    if "$" in t:
        return True

    for trig in _LATEX_TRIGGERS:
        if trig in t:
            return True

    if re.search(r"[\^_]", t):
        return True
    if re.search(r"[=<>]", t):
        return True

    if re.search(r"\b(s|t|x|y|r)\b", t) and re.search(r"\d", t):
        return True

    return False


def _ensure_math_wrapped_if_needed(text: str) -> str:
    """
    Streamlit renders LaTeX only inside $...$ (in markdown).
    If the text is mathy and not already wrapped, wrap the WHOLE expression.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if "$" in t:
        return t
    return f"${t}$" if _looks_like_math(t) else t


def _normalize_math_text(text: str) -> str:
    """
    Make math display-friendly for Streamlit:
    - Collapse double-escaped backslashes (\\frac -> \frac)
    - Convert \( \) and \[ \] to $...$
    - Best-effort convert short a/b-style fractions to \frac{a}{b}
    """
    if not text:
        return ""

    out = str(text)

    # If stored as "\\frac{1}{s}", display needs "\frac{1}{s}"
    out = out.replace("\\\\", "\\")

    # Normalize \(...\) and \[...\] to $...$
    out = re.sub(r"\\\((.*?)\\\)", r"$\1$", out, flags=re.DOTALL)
    out = re.sub(r"\\\[(.*?)\\\]", r"$\1$", out, flags=re.DOTALL)

    # Replace common "a/b" style with \frac{a}{b} (conservative)
    if "/" in out and "\\" not in out and "$" not in out:
        if len(out.strip()) <= 80 and re.search(r"[A-Za-z0-9\)\]]\s*/\s*[\(\[]", out):
            out2 = re.sub(
                r"^\s*\(?\s*([A-Za-z0-9\^\_\+\-\s\(\)]+)\s*\)?\s*/\s*\(?\s*([A-Za-z0-9\^\_\+\-\s\(\)]+)\s*\)?\s*$",
                r"\\frac{\1}{\2}",
                out.strip(),
            )
            out = out2

    # Optional extra normalizations
    out = re.sub(r"\bL\{([^}]+)\}", r"\\mathcal{L}{\{\1\}}", out)
    out = re.sub(r"\bsqrt\(([^)]+)\)", r"\\sqrt{\1}", out)

    return out.strip()


def _format_choice_like_quiz_engine(s: str) -> str:
    """
    Keep labels like 'A) ' outside math, but make the remainder renderable.
    """
    raw = (s or "").strip()
    if not raw:
        return ""

    m = re.match(r"^([ABCD])\s*[\)\.\-:]\s*(.*)$", raw, flags=re.IGNORECASE)
    if not m:
        cleaned = _normalize_math_text(raw)
        return _ensure_math_wrapped_if_needed(cleaned)

    lab = m.group(1).upper() + ") "
    rest = (m.group(2) or "").strip()
    rest = _normalize_math_text(rest)
    rest = _ensure_math_wrapped_if_needed(rest)
    return lab + rest


# ------------------ INIT ------------------
def init_page():
    init_stats_db()
    user_id = st.session_state.get("user_id")
    return DBManager(backend="sqlite", user_id=user_id, db_file=CHAT_DB_FILE)


db = init_page()

# ------------------ SUBJECT SELECTION ------------------
st.title("📊 Progress & Statistics")

# ✅ FILTER OUT "General" from selectable subjects
subject_names = [s for s in (db.load_all_subjects() or []) if s != "General"]

if not subject_names:
    st.warning("⚠️ No subjects found (excluding 'General'). Create a subject in the main app first.")
    st.stop()

# If current_subject is missing/invalid/General, default to first real subject
current_default = st.session_state.get("current_subject")
if not current_default or current_default == "General" or current_default not in subject_names:
    current_default = subject_names[0]

selected_subject = st.selectbox(
    "📚 Select Subject to Analyze",
    subject_names,
    index=subject_names.index(current_default),
)

if selected_subject != st.session_state.get("current_subject"):
    st.session_state.current_subject = selected_subject
    st.rerun()

# ------------------ DATA RETRIEVAL ------------------
attempts = get_attempts_for_subject(selected_subject, user_id=st.session_state.get("user_id"))

if not attempts:
    st.info(f"No attempts recorded for **{selected_subject}** yet.")
    st.stop()

# ------------------ RESET ------------------
with st.expander("⚠️ Danger Zone"):
    if st.button(f"🗑️ Reset all statistics for {selected_subject}"):
        clear_subject_stats(selected_subject)
        st.success(f"Cleared all statistics for {selected_subject}.")
        st.rerun()

# ------------------ DATAFRAME ------------------
df = pd.DataFrame(attempts)

# ✅ remove source_hint completely (you requested this)
if "source_hint" in df.columns:
    df = df.drop(columns=["source_hint"])

cols = ["ts_ms", "is_correct", "topic", "question", "chosen", "correct", "difficulty"]
for col in cols:
    if col not in df.columns:
        df[col] = None

# ---- ts_ms cleaning (DON'T DROP if missing/0; FIX it) ----
df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce")

now_ms = int(datetime.now().timestamp() * 1000)
df["ts_ms"] = df["ts_ms"].fillna(0)

df.loc[df["ts_ms"] <= 0, "ts_ms"] = pd.NA
df["ts_ms"] = df["ts_ms"].ffill()
df["ts_ms"] = df["ts_ms"].fillna(now_ms).astype("int64")

# ---- only drop truly empty questions ----
df["question"] = df["question"].astype(str)
df = df[df["question"].str.strip().ne("")].copy()

# stable sort so equal timestamps don't reorder unpredictably
df = df.sort_values(["ts_ms"], kind="mergesort").reset_index(drop=True)

if df.empty:
    st.warning("No valid data points found.")
    st.stop()

df["datetime"] = df["ts_ms"].apply(lambda ms: datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M"))
df["correct_bool"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int) == 1


# Handle Topic Fallbacks
def get_clean_topic(row):
    t = str(row.get("topic") or "").strip()
    if t and t.lower() != "none":
        return t
    diff = str(row.get("difficulty") or "").strip()
    return f"General ({diff})" if diff else "General"


df["topic"] = df.apply(get_clean_topic, axis=1)

# ✅ Auto-merge similar topics (NO hard-coded replacements)
alias = build_topic_alias_map(df["topic"].astype(str).tolist(), threshold=0.88)
df["topic"] = df["topic"].astype(str).map(lambda x: alias.get(x, x))

# Attempt grouping
df["q_index"] = df.index + 1
df["attempt_id"] = ((df["q_index"] - 1) // QUESTIONS_PER_ATTEMPT) + 1
df["q_in_attempt"] = ((df["q_index"] - 1) % QUESTIONS_PER_ATTEMPT) + 1

# ------------------ DISPLAY-FRIENDLY (LaTeX) COLUMNS ------------------
df["question_disp"] = df["question"].map(lambda x: _normalize_math_text(str(x)))
df["chosen_disp"] = df["chosen"].map(lambda x: _format_choice_like_quiz_engine(str(x)))
df["correct_disp"] = df["correct"].map(lambda x: _format_choice_like_quiz_engine(str(x)))

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

    chosen_id = st.selectbox("View Attempt #", options=available_ids, index=len(available_ids) - 1)

# ------------------ METRICS DASHBOARD ------------------
df_scope = df if show_inc else df[df["attempt_id"].isin(complete_ids)]
att_df = df[df["attempt_id"] == chosen_id]

total_q, total_correct = len(df_scope), int(df_scope["correct_bool"].sum())
att_q, att_correct = len(att_df), int(att_df["correct_bool"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Questions", total_q)
m2.metric("Overall Accuracy", f"{(total_correct / total_q * 100):.1f}%" if total_q else "0%")
m3.metric(f"Attempt {chosen_id} Score", f"{att_correct} / {att_q}")
m4.metric(f"Attempt {chosen_id} Accuracy", f"{(att_correct / att_q * 100):.1f}%" if att_q else "0%")

st.divider()

# ------------------ TOPIC ANALYSIS ------------------
st.subheader(f"🏷️ Topic Breakdown: Attempt {chosen_id}")
topic_stats = att_df.groupby("topic").agg(
    Questions=("question", "count"),
    Correct=("correct_bool", "sum"),
).reset_index()
topic_stats["Accuracy %"] = (topic_stats["Correct"] / topic_stats["Questions"] * 100).round(1)

st.dataframe(topic_stats.sort_values("Accuracy %"), width='stretch', hide_index=True)

# ------------------ ATTEMPT HISTORY ------------------
st.subheader("🧾 Score History")
history = df_scope.groupby("attempt_id").agg(
    Date=("datetime", "min"),
    Questions=("question", "count"),
    Correct=("correct_bool", "sum"),
).reset_index()

history["Score"] = history.apply(lambda r: f"{int(r['Correct'])} / {int(r['Questions'])}", axis=1)
history["Accuracy %"] = (history["Correct"] / history["Questions"] * 100).round(1)

st.dataframe(history.sort_values("attempt_id", ascending=False), width='stretch', hide_index=True)

# ------------------ MISTAKE TRACKER ------------------
st.subheader("❌ Recent Mistakes")
mistakes = df_scope[df_scope["correct_bool"] == False].sort_values("ts_ms", ascending=False).head(30)

if mistakes.empty:
    st.success("No mistakes found! Keep it up.")
else:
    render_math = st.toggle("Detailed View", value=True)

    if not render_math:
        view_cols = ["datetime", "attempt_id", "topic", "question", "chosen", "correct"]
        st.dataframe(
            mistakes[view_cols].rename(
                columns={
                    "datetime": "Time",
                    "attempt_id": "Attempt",
                    "chosen": "Your Answer",
                    "correct": "Correct Answer",
                }
            ),
            width='stretch',
            hide_index=True,
        )
    else:
        for _, r in mistakes.iterrows():
            time_txt = str(r.get("datetime", ""))
            att_id = str(r.get("attempt_id", ""))
            topic = str(r.get("topic", ""))

            q = str(r.get("question_disp", r.get("question", "")))
            chosen = str(r.get("chosen_disp", r.get("chosen", "")))
            correct = str(r.get("correct_disp", r.get("correct", "")))

            with st.expander(f"{time_txt}  |  Attempt {att_id}  |  {topic}"):
                st.markdown("**Question:**")
                st.markdown(q)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Your Answer:**")
                    st.markdown(chosen)
                with c2:
                    st.markdown("**Correct Answer:**")
                    st.markdown(correct)

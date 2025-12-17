from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import ollama
from openai import OpenAI

# NEW: use the subject index so we know which slide/page a question came from
from modules.quizstats.file_utils import get_subject_index_entries


# -------------------------------------------------------------
#   OLLAMA/OPENAI CLIENT HANDLING
# -------------------------------------------------------------
def _get_openai_client() -> OpenAI | None:
    """Gets the OpenAI client if the API key is available."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


# -------------------------------------------------------------
#   FALLBACK CHUNKING (used only if index is missing)
# -------------------------------------------------------------
def _split_corpus_into_chunks(corpus_text: str, max_chars: int = 1800) -> List[str]:
    """
    Old behaviour: split the plain corpus into text chunks.
    Only used as a fallback if the page_index.json is missing.
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n", corpus_text) if p.strip()]
    scored: List[tuple[int, str]] = []

    for p in parts:
        length = len(p)
        if length < 120:
            continue

        score = 1

        q_markers = [
            r"Question\s*\d+",
            r"Q\d+",
            r"\bchoose\b",
            r"\bwhich\b",
            r"\bbased on\b",
            r"\brefer to\b",
        ]
        for pat in q_markers:
            if re.search(pat, p, flags=re.IGNORECASE):
                score += 3

        if re.search(r"Figure|Chart|Diagram|Table|Histogram|Scatter", p, flags=re.IGNORECASE):
            score += 4

        if length > 600:
            score += 2

        if length > max_chars:
            p = p[: max_chars // 2] + "\n...\n" + p[-max_chars // 2 :]

        scored.append((score, p))

    random.shuffle(scored)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


# -------------------------------------------------------------
#   INDEX-AWARE CHUNK BUILDER
# -------------------------------------------------------------
def _build_chunks_from_index(subject: str, max_chars: int = 1800) -> List[Dict[str, Any]]:
    """
    Build chunks directly from page_index.json so we know
    exactly which slide/page each chunk comes from.

    Returns:
        [{"text": "...", "context": <index entry dict>}]

    We keep at most ONE chunk per index entry so each slide/page can yield
    at most one question.
    """
    entries = get_subject_index_entries(subject)
    chunks: List[Dict[str, Any]] = []

    for entry in entries:
        text = (entry.get("text") or "").strip()
        if len(text) < 80:
            continue

        if len(text) > max_chars:
            text = text[: max_chars // 2] + "\n...\n" + text[-max_chars // 2 :]

        chunks.append({"text": text, "context": entry})

    return chunks


# -------------------------------------------------------------
#   OLLAMA QUESTION GENERATOR
# -------------------------------------------------------------
def _call_ollama_for_chunk(
    model_name: str,
    chunk_text: str,
    subject: str,
    max_q: int,
    context_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generates quiz questions using a local Ollama model."""

    system_msg = (
        "You are an expert exam question writer for a polytechnic module.\n"
        "Your job is to create challenging, contextualised multiple-choice questions.\n"
        "Use ONLY the provided notes; do not invent new data.\n"
        "If the text references images/figures/tables, set `needs_image: true`.\n"
        "Return a single JSON object ONLY (no markdown)."
    )

    user_prompt = f"""
Generate up to {max_q} challenging MCQs from the notes below.

Notes (from subject: {subject}):
<notes>
{chunk_text}
</notes>

RULES:
- Questions must test concepts, not just copy numbers/labels.
- EXACT 4 options.
- EXACT 1 correct answer.
- Include a short explanation.
- Include difficulty: "easy", "medium", or "hard".
- Include a short topic label (2–6 words) describing the concept (e.g. "Z-score calculation", "Scatter plots and correlation").
- If notes refer to chart/table/figure/workflow, set:
    "needs_image": true
    "image_hint": phrase to locate this slide/page
  else needs_image false.

Return JSON ONLY in this format:
{{
  "questions": [
    {{
      "question": "text (you may include LaTeX like $\\\\sigma = \\\\sqrt{{...}}$)",
      "choices": ["A", "B", "C", "D"],
      "answer_index": 0,
      "explanation": "...",
      "needs_image": false,
      "image_hint": "",
      "difficulty": "medium",
      "topic": "Short topic here"
    }}
  ]
}}
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.5,
                "num_predict": 4096,
            },
            format="json",
        )
        raw = response["message"]["content"]
    except Exception as e:
        print(f"[quiz_engine] Ollama error ({model_name}):", e)
        return []

    return _parse_raw_quiz_json(raw, chunk_text, context_meta)


# -------------------------------------------------------------
#   OPENAI QUESTION GENERATOR
# -------------------------------------------------------------
def _call_openai_for_chunk(
    model_name: str,
    chunk_text: str,
    subject: str,
    max_q: int,
    context_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generates quiz questions using the OpenAI API."""

    client = _get_openai_client()
    if client is None:
        print("[quiz_engine] OpenAI API key missing inside chunk call. Skipping.")
        return []

    system_msg = (
        "You are an expert exam question writer for a polytechnic module.\n"
        "Your job is to create challenging, contextualised multiple-choice questions.\n"
        "Use ONLY the provided notes; do not invent new data.\n"
        "Return JSON only."
    )

    user_prompt = f"""
Generate up to {max_q} challenging MCQs from the notes below.

Notes (from subject: {subject}):
<notes>
{chunk_text}
</notes>

RULES:
- Questions must test concepts, not just copy numbers/labels.
- EXACT 4 options.
- EXACT 1 correct answer.
- Include a short explanation.
- Include difficulty: "easy", "medium", or "hard".
- Include a short topic label (2–6 words) describing the concept (e.g. "Z-score calculation", "Scatter plots and correlation").
- If notes refer to chart/table/figure/workflow, set:
    "needs_image": true
    "image_hint": phrase to locate this slide/page
  else needs_image false.

Return JSON ONLY in this format:
{{
  "questions": [
    {{
      "question": "text (you may include LaTeX like $\\\\sigma = \\\\sqrt{{...}}$)",
      "choices": ["A", "B", "C", "D"],
      "answer_index": 0,
      "explanation": "...",
      "needs_image": false,
      "image_hint": "",
      "difficulty": "medium",
      "topic": "Short topic here"
    }}
  ]
}}
"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.35,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            # Note: OpenAI's client handles JSON response formatting separately from chat completions
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print("[quiz_engine] OpenAI error:", e)
        return []

    return _parse_raw_quiz_json(raw, chunk_text, context_meta)


# -------------------------------------------------------------
#   JSON PARSING + TOPIC GUARANTEE
# -------------------------------------------------------------
def _parse_raw_quiz_json(
    raw: str,
    chunk_text: str,
    context_meta: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if not match:
        print("[quiz_engine] JSON block missing.")
        return []
    json_text = match.group(0)

    try:
        data = json.loads(json_text)
    except Exception:
        print("[quiz_engine] JSON parse error.")
        return []

    qs = data.get("questions", []) or []
    out: List[Dict[str, Any]] = []

    for q in qs:
        question = str(q.get("question", "")).strip()
        choices = [str(c).strip() for c in (q.get("choices", []) or [])]
        if len(choices) != 4 or not question:
            continue

        try:
            ans = int(q.get("answer_index", 0))
        except Exception:
            ans = 0
        if ans < 0 or ans > 3:
            ans = 0

        explanation = str(q.get("explanation", "")).strip() or "Refer to notes."
        needs_image = bool(q.get("needs_image", False))
        image_hint = str(q.get("image_hint", "")).strip()
        difficulty = str(q.get("difficulty", "medium")).strip() or "medium"

        # ---- TOPIC (NO MORE 'No topic') ----
        topic = str(q.get("topic", "")).strip()

        # 1) If model didn't provide, use image_hint
        if not topic:
            topic = image_hint.strip()

        # 2) If still empty, derive from context (page_index.json)
        ctx = context_meta or {}
        fname = (ctx.get("file") or "").strip()
        ctype = (ctx.get("type") or "").strip()

        if not topic and fname:
            if ctype == "pptx_slide" and ctx.get("slide"):
                topic = f"{fname} – Slide {ctx.get('slide')}"
            elif ctype == "pdf_page" and ctx.get("page"):
                topic = f"{fname} – Page {ctx.get('page')}"
            else:
                topic = fname

        # 3) Absolute last fallback
        if not topic:
            topic = "General"

        # Minimal source snippet
        src_hint = " ".join(chunk_text.split()[:30])

        # Attach context metadata for “show context”
        context_info = {
            "type": ctx.get("type"),
            "file": ctx.get("file"),
            "file_path": ctx.get("file_path"),
            "page": ctx.get("page"),
            "slide": ctx.get("slide"),
        }

        out.append(
            {
                "question": question,
                "choices": choices,
                "answer_idx": ans,
                "explanation": explanation,
                "needs_image": needs_image,
                "image_hint": image_hint,
                "source_hint": src_hint,
                "difficulty": difficulty,
                "topic": topic,  # ✅ ALWAYS NON-EMPTY
                "context_meta": context_info,
            }
        )

    return out


# -------------------------------------------------------------
#   FALLBACK (NO LLM)
# -------------------------------------------------------------
def _fallback_quiz(corpus_text: str, subject: str, n_questions: int) -> List[Dict[str, Any]]:
    sents = re.split(r"(?<=[.!?])\s+", corpus_text)
    sents = [s for s in sents if 60 < len(s) < 240]

    random.shuffle(sents)
    out: List[Dict[str, Any]] = []

    for s in sents[:n_questions]:
        words = s.split()
        if len(words) < 6:
            continue
        idx = random.randint(3, len(words) - 3)
        ans = words[idx]
        words[idx] = "_____"
        stem = " ".join(words)

        wrongs = random.sample([w for w in words if w != "_____"], min(3, max(3, len(words) - 1)))
        choices = [ans] + wrongs
        random.shuffle(choices)

        out.append(
            {
                "question": stem,
                "choices": choices[:4] if len(choices) >= 4 else (choices + ["..."])[:4],
                "answer_idx": choices.index(ans) if ans in choices else 0,
                "explanation": "Look at the original sentence.",
                "needs_image": False,
                "image_hint": "",
                "source_hint": s[:40],
                "difficulty": "easy",
                "topic": "General",  # ✅ fallback still has topic
                "context_meta": {},
            }
        )

    return out


# -------------------------------------------------------------
#   MAIN PUBLIC API
# -------------------------------------------------------------
def build_quiz_from_corpus(
    corpus_text: str,
    subject: str,
    model_name: str,
    n_questions: int = 10,
) -> List[Dict[str, Any]]:

    corpus_text = (corpus_text or "").strip()
    if not corpus_text:
        return []

    is_openai_model = "gpt" in model_name.lower()
    
    # The call function now only takes 5 arguments (model_name, chunk_text, etc.)
    call_fn = _call_openai_for_chunk if is_openai_model else _call_ollama_for_chunk
    
    # If it's an OpenAI model, the call function internally checks the API key.
    # If it's an Ollama model, the call function internally handles the connection.

    indexed_chunks = _build_chunks_from_index(subject)
    quiz: List[Dict[str, Any]] = []

    if indexed_chunks:
        random.shuffle(indexed_chunks)

        # ONE question per index entry => reduces repeats
        for item in indexed_chunks:
            remain = n_questions - len(quiz)
            if remain <= 0:
                break

            chunk_text = item["text"]
            ctx = item["context"]

            # The call_fn signature is now consistent: (model_name, chunk_text, subject, max_q, context_meta)
            qs = call_fn(
                model_name,
                chunk_text,
                subject,
                max_q=1,
                context_meta=ctx,
            )

            for q in qs:
                # prevent identical question text duplicates
                if all(q["question"] != e["question"] for e in quiz):
                    quiz.append(q)
                    break

    # Top-up if needed
    if len(quiz) < n_questions:
        print("[quiz_engine] Index produced fewer questions; topping up.")
        plain_chunks = _split_corpus_into_chunks(corpus_text)
        random.shuffle(plain_chunks)

        for chunk in plain_chunks:
            remain = n_questions - len(quiz)
            if remain <= 0:
                break

            # The call_fn signature is now consistent: (model_name, chunk, subject, max_q, context_meta)
            qs = call_fn(
                model_name,
                chunk,
                subject,
                max_q=min(3, remain),
                context_meta=None,
            )

            for q in qs:
                if all(q["question"] != e["question"] for e in quiz):
                    quiz.append(q)
                    if len(quiz) >= n_questions:
                        break

    if not quiz:
        # If no questions were generated (e.g., LLM errors, network problems), fall back
        return _fallback_quiz(corpus_text, subject, n_questions)

    return quiz

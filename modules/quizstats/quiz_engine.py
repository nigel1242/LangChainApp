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
    """Generates quiz questions using a local Ollama model with strict source grounding."""

    system_msg = (
        "You are a strict academic examiner. Your priority is FACTUAL ACCURACY based ONLY on the provided notes.\n"
        "1. Identify concepts ONLY found in the <notes>. Do NOT use general knowledge or outside software facts.\n"
        "2. Create a question and 4 distinct options.\n"
        "3. VERIFY: Ensure the 'answer_index' correctly matches the correct choice based on the notes.\n"
        "4. Use LaTeX for math: enclose in $...$ and use double backslashes (\\\\).\n"
        "Return valid JSON only."
    )

    user_prompt = f"""
Generate up to {max_q} challenging MCQs from the notes below for the subject: {subject}.

<notes>
{chunk_text}
</notes>

STRICT RULES:
- Use ONLY the provided notes.
- Do NOT invent data. For example, if Tableau is not in the notes, do not ask about it.
- In the 'explanation', quote the specific sentence from the notes that supports the answer.
- Ensure 'answer_index' (0-3) is 100% accurate.

JSON format:
{{
  "questions": [
    {{
      "question": "text",
      "choices": ["A", "B", "C", "D"],
      "answer_index": 0,
      "explanation": "Fact Check: According to the notes, [Quote/Explanation]...",
      "needs_image": false,
      "image_hint": "",
      "difficulty": "medium",
      "topic": "Topic Label"
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
                "temperature": 0.1,  # Lowered for higher strictness
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
    """Generates quiz questions using OpenAI with high-fidelity checks."""

    client = _get_openai_client()
    if client is None: return []

    system_msg = (
        "You are a strict academic examiner. Your sole source of truth is the <notes> provided.\n"
        "1. Do NOT use outside knowledge (e.g., general software facts) unless explicitly mentioned in the notes.\n"
        "2. If the notes discuss 'Measure Names' in Tableau, use that. If they don't, do not invent the question.\n"
        "3. Every question must have a 'Verified' explanation that quotes or paraphrases the notes."
    )

    user_prompt = f"""
Generate up to {max_q} MCQs from these notes.

<notes>
{chunk_text}
</notes>

MANDATORY CHECKS:
1. Is the question testing a specific concept?
2. Does the 'answer_index' (0-3) actually point to the correct answer?
3. In the explanation, explicitly state: 'Fact check: [Concept] is defined as [Definition].'

Return JSON ONLY:
{{
  "questions": [
    {{
      "question": "...",
      "choices": ["...", "...", "...", "..."],
      "answer_index": 0,
      "explanation": "...",
      "needs_image": false,
      "image_hint": "",
      "difficulty": "hard",
      "topic": "..."
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

def build_quiz_from_rag(
    subject: str,
    model_name: str,
    n_questions: int = 10,
    vector_db: str = "faiss"
) -> List[Dict[str, Any]]:
    from modules.functions import get_relevant_rag
    import random

    # 1. PATH LOGIC: Try both 'Dava' and 'subject_Dava'
    possible_ids = [f"subject_{subject}", subject]
    rag_content = ""
    has_docs = False

    # Dispatcher setup
    try:
        raw_ollama = [m["model"] for m in ollama.list().get("models", [])]
        ollama_models = tuple(raw_ollama)
    except:
        ollama_models = ()
    openai_models = ("gpt-4o-mini", "gpt-4o", "gpt-4")

    # 2. RETRIEVAL LOOP: Find which folder actually exists
    for tid in possible_ids:
        content, found = get_relevant_rag(
            db_file="chat_playground.db",
            rag_index_dir="rag_indices",
            target_id=tid,
            query=f"Overview of {subject}",
            model_name=model_name,
            ollama_models=ollama_models,
            openai_models=openai_models,
            top_k=15,
            vector_db=vector_db
        )
        if found and content.strip():
            rag_content = content
            has_docs = True
            break # Found the folder!

    if not has_docs:
        print(f"[quiz_engine] Critical: No FAISS index found for {subject}")
        return []

    # 3. GENERATION
    is_openai = "gpt" in model_name.lower()
    call_fn = _call_openai_for_chunk if is_openai else _call_ollama_for_chunk
    
    chunks = _split_corpus_into_chunks(rag_content)
    random.shuffle(chunks)
    quiz = []

    for chunk in chunks:
        if len(quiz) >= n_questions: break
        
        try:
            qs = call_fn(model_name, chunk, subject, max_q=1, context_meta=None)
            for q in qs:
                if all(q["question"] != e["question"] for e in quiz):
                    quiz.append(q)
        except Exception as e:
            print(f"LLM Error: {e}")
            continue

    return quiz
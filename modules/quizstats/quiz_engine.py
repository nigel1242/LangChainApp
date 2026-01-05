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
#   PROMPTS
# -------------------------------------------------------------
def _get_unified_system_msg():
    return (
        "You are a strict academic examiner. Your sole source of truth is the <notes> provided.\n"
        "1. Identify concepts ONLY found in the <notes>. Do NOT use outside knowledge or general facts.\n"
        "2. Create a question and exactly 4 distinct options. \n"
        "3. FORMAT: You MUST prefix each choice with 'A) ', 'B) ', 'C) ', and 'D) '.\n"
        "4. VERIFY: Ensure the 'answer_index' correctly matches the correct choice (0=A, 1=B, 2=C, 3=D).\n"
        "5. EXPLANATION: Quote the specific sentence from the notes that supports the answer.\n"
        "6. Use LaTeX for math: enclose in $...$ and use double backslashes (\\\\).\n"
        "Return valid JSON only."
    )

def _get_unified_user_prompt(max_q, subject, chunk_text):
    return f"""
Generate up to {max_q} challenging MCQs from the notes below for the subject: {subject}.

<notes>
{chunk_text}
</notes>

STRICT RULES:
- Use ONLY the provided notes. Do NOT invent data.
- Choices MUST be formatted as: ["A) text", "B) text", "C) text", "D) text"]
- Ensure 'answer_index' (0-3) is 100% accurate.

JSON format:
{{
  "questions": [
    {{
      "question": "text",
      "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer_index": 0,
      "explanation": "Fact Check: According to the notes, [Quote]...",
      "needs_image": false,
      "image_hint": "",
      "difficulty": "medium",
      "topic": "Topic Label"
    }}
  ]
}}
"""

# -------------------------------------------------------------
#   OLLAMA QUESTION GENERATOR
# -------------------------------------------------------------
def _call_ollama_for_chunk(model_name, chunk_text, subject, max_q, context_meta=None):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": _get_unified_system_msg()},
                {"role": "user", "content": _get_unified_user_prompt(max_q, subject, chunk_text)},
            ],
            options={"temperature": 0.1, "num_predict": 4096},
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
def _call_openai_for_chunk(model_name, chunk_text, subject, max_q, context_meta=None):
    client = _get_openai_client()
    if client is None: return []

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.1, # Set to match Ollama's strictness
            messages=[
                {"role": "system", "content": _get_unified_system_msg()},
                {"role": "user", "content": _get_unified_user_prompt(max_q, subject, chunk_text)},
            ],
            response_format={"type": "json_object"} # Forces JSON mode for OpenAI
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

def build_quiz_from_rag(subject, model_name, n_questions: int = 10, vector_db: str = "faiss") -> List[Dict[str, Any]]:
    from modules.functions import get_relevant_rag
    import os
    import random

    # 1. NEW PATH LOGIC: Point directly to the subject folder
    subject_dir = os.path.join("modules", "subjects", subject)
    
    # 2. SUFFIX LOGIC: Match the index naming convention
    is_openai = "gpt" in model_name.lower()
    suffix = "openai" if is_openai else "ollama"

    # Setup model lists for the dispatcher
    try:
        import ollama
        raw_ollama = [m["model"] for m in ollama.list().get("models", [])]
        ollama_models = tuple(raw_ollama)
    except:
        ollama_models = ()
    openai_models = ("gpt-4o-mini", "gpt-4o", "gpt-4")

    # 3. RETRIEVAL: Using the updated directory and suffix
    # Note: Added 'suffix' parameter to match your functions.py logic
    rag_content, has_docs = get_relevant_rag(
        db_file="chat_playground.db",
        rag_index_dir=subject_dir, # Looking in modules/subjects/Dava/
        target_id=subject,
        query=f"Generate quiz questions about {subject}",
        model_name=model_name,
        ollama_models=ollama_models,
        openai_models=openai_models,
        top_k=15,
        vector_db=vector_db,
        suffix=suffix # This ensures it opens index_ollama or index_openai
    )

    if not has_docs or not rag_content.strip():
        print(f"[quiz_engine] Critical: No {suffix} index found for {subject} at {subject_dir}")
        return []

    # 4. GENERATION
    call_fn = _call_openai_for_chunk if is_openai else _call_ollama_for_chunk
    
    # Split the retrieved text into manageable pieces for the LLM
    chunks = _split_corpus_into_chunks(rag_content)
    random.shuffle(chunks)
    quiz = []

    for chunk in chunks:
        if len(quiz) >= n_questions: 
            break
        
        try:
            # Generate 1-2 questions per chunk to ensure variety
            qs = call_fn(model_name, chunk, subject, max_q=2) 
            for q in qs:
                # Basic duplicate prevention
                if all(q["question"] != e["question"] for e in quiz):
                    quiz.append(q)
        except Exception as e:
            print(f"LLM Generation Error for chunk: {e}")
            continue

    return quiz[:n_questions]
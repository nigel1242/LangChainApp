# modules/quizstats/quiz_engine.py
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
#   QUALITY HELPERS (Blueprint + Verify)
#   (kept in this same file; no extra engine)
# -------------------------------------------------------------
def _extract_json_block(raw: str) -> dict | None:
    if not raw:
        return None
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def _llm_json(model_name: str, system: str, user: str) -> dict | None:
    """Unified JSON call for OpenAI/Ollama. Returns dict or None."""
    is_openai_model = "gpt" in (model_name or "").lower()

    if is_openai_model:
        client = _get_openai_client()
        if client is None:
            return None
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            return _extract_json_block(raw)
        except Exception:
            return None

    # Ollama
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.2, "num_predict": 2048},
            format="json",
        )
        raw = resp["message"]["content"]
        return _extract_json_block(raw)
    except Exception:
        return None


def _blueprint_system() -> str:
    return (
        "You are an expert educator. You ONLY use the provided notes.\n"
        "Extract a topic blueprint that will be used to generate exam-quality MCQs.\n"
        "Return JSON only."
    )


def _blueprint_user(subject: str, notes: str) -> str:
    return f"""
Create a topic blueprint for the subject: {subject}

<notes>
{notes}
</notes>

Return JSON in exactly this schema:
{{
  "topics": [
    {{
      "topic": "Short topic name",
      "question_styles": ["definition", "application", "calculation", "compare-contrast", "common-misconception", "graph-interpretation", "inference"],
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ]
}}

Rules:
- topics must come ONLY from the notes
- produce 6 to 12 topics (if possible)
- keywords should be directly searchable terms in the notes
"""


def _verify_system() -> str:
    return (
        "You are a verifier. You ONLY use the <notes>.\n"
        "Check if the MCQ is valid and supported by the notes.\n"
        "Return JSON only."
    )


def _verify_user(notes: str, mcq_json: dict) -> str:
    return f"""
<notes>
{notes}
</notes>

<mcq>
{json.dumps(mcq_json, ensure_ascii=False)}
</mcq>

Return JSON:
{{
  "ok": true/false,
  "reason": "short reason",
  "fixed": {{
    "question": "...",
    "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer_index": 0,
    "explanation": "Quote: '...'(from notes). Reasoning: ...",
    "difficulty": "easy|medium|hard",
    "topic": "..."
  }}
}}

Rules:
- ok=true only if the correct answer is directly supported by the notes AND the explanation contains a direct quote.
- The explanation MUST include a short quote + reasoning/inference/calculation (not just a quote).
- If minor issues exist (format, answer_index mismatch), fix it in 'fixed'.
- If unsupported/hallucinated, ok=false.
"""


# -------------------------------------------------------------
#   FALLBACK CHUNKING (used only if index is missing)
# -------------------------------------------------------------
def _split_corpus_into_chunks(corpus_text: str, max_chars: int = 1800) -> List[str]:
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
#   PROMPTS (UPDATED: high-insight + banned NOT/EXCEPT)
# -------------------------------------------------------------
def _get_unified_system_msg():
    return (
        "You are a strict academic examiner and learning coach.\n"
        "Your ONLY source of truth is the provided <notes>.\n\n"
        "GOAL: Create HIGH-INSIGHT MCQs that help the student learn.\n"
        "Allowed question styles (prefer these):\n"
        "- calculation / compute from values in the notes\n"
        "- interpretation of a chart/graph/table/figure described in the notes\n"
        "- inference (what can be concluded, what changes if..., why..., which method suits...)\n"
        "- application (choose the best method/step/approach for a scenario)\n"
        "- compare/contrast (difference between two concepts in the notes)\n\n"
        "BANNED question styles (do NOT generate these):\n"
        "- 'Which of the following is NOT...' / 'EXCEPT' / 'NOT listed' / 'not stated'\n"
        "- pure recall of links/references/metadata/pages\n"
        "- trivial purpose questions like 'What is the purpose of this reading material?'\n\n"
        "RULES:\n"
        "1) Use ONLY what is explicitly present in <notes>. No outside knowledge.\n"
        "2) Create exactly 1 question and exactly 4 distinct options.\n"
        "3) Choices MUST be prefixed 'A) ', 'B) ', 'C) ', 'D) '.\n"
        "4) Ensure answer_index is correct (0=A, 1=B, 2=C, 3=D).\n"
        "5) Explanation MUST include:\n"
        "   - a short direct quote from the notes AND\n"
        "   - 1–2 sentences of reasoning/inference/calculation (not just restating)\n"
        "6) Use LaTeX for math ($...$) and escape backslashes (\\\\).\n"
        "Return valid JSON only."
    )


def _get_unified_user_prompt(max_q, subject, chunk_text):
    return f"""
Generate up to {max_q} exam-quality MCQs from the notes below for subject: {subject}.

<notes>
{chunk_text}
</notes>

IMPORTANT QUALITY REQUIREMENTS:
- Do NOT ask 'NOT/EXCEPT/not stated/not listed' questions.
- Do NOT ask questions about references/URLs or document housekeeping.
- Prefer calculation, graph/table interpretation, inference, application, compare/contrast.
- The question must require thinking, not just scanning for a phrase.

JSON format:
{{
  "questions": [
    {{
      "question": "text",
      "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer_index": 0,
      "explanation": "Quote: '...'(from notes). Reasoning: ...",
      "needs_image": false,
      "image_hint": "",
      "difficulty": "easy|medium|hard",
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
    if client is None:
        return []

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _get_unified_system_msg()},
                {"role": "user", "content": _get_unified_user_prompt(max_q, subject, chunk_text)},
            ],
            response_format={"type": "json_object"},
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

        topic = str(q.get("topic", "")).strip()
        if not topic:
            topic = image_hint.strip()

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

        if not topic:
            topic = "General"

        src_hint = " ".join(chunk_text.split()[:30])

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
                "topic": topic,
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
                "topic": "General",
                "context_meta": {},
            }
        )

    return out


# -------------------------------------------------------------
#   MAIN PUBLIC API (CORPUS)
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
    call_fn = _call_openai_for_chunk if is_openai_model else _call_ollama_for_chunk

    indexed_chunks = _build_chunks_from_index(subject)
    quiz: List[Dict[str, Any]] = []

    if indexed_chunks:
        random.shuffle(indexed_chunks)

        for item in indexed_chunks:
            remain = n_questions - len(quiz)
            if remain <= 0:
                break

            chunk_text = item["text"]
            ctx = item["context"]

            qs = call_fn(
                model_name,
                chunk_text,
                subject,
                max_q=1,
                context_meta=ctx,
            )

            for q in qs:
                if all(q["question"] != e["question"] for e in quiz):
                    quiz.append(q)
                    break

    if len(quiz) < n_questions:
        print("[quiz_engine] Index produced fewer questions; topping up.")
        plain_chunks = _split_corpus_into_chunks(corpus_text)
        random.shuffle(plain_chunks)

        for chunk in plain_chunks:
            remain = n_questions - len(quiz)
            if remain <= 0:
                break

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
        return _fallback_quiz(corpus_text, subject, n_questions)

    return quiz


# -------------------------------------------------------------
#   MAIN PUBLIC API (RAG)  ✅ DIVERSITY + NO LOW-VALUE QUESTIONS
# -------------------------------------------------------------
def build_quiz_from_rag(subject, model_name, n_questions: int = 10, vector_db: str = "faiss") -> List[Dict[str, Any]]:
    """
    Single-engine flow:
    - Build blueprint topics
    - Retrieve per-topic contexts
    - Generate 1 question per new context
    - Verify + fix
    - Hard filters:
        * ban NOT/EXCEPT / "not stated" / reference/link questions
        * ban trivial purpose/housekeeping questions
        * require explanation includes quote + reasoning
        * avoid repeated context (file/page/slide)
        * avoid rephrased duplicates (near-duplicate detection)
    """
    from modules.functions import get_relevant_rag
    from difflib import SequenceMatcher

    subject_dir = os.path.join("modules", "subjects", subject)
    is_openai = "gpt" in model_name.lower()
    suffix = "openai" if is_openai else "ollama"

    # -------------------------
    # LOW-VALUE FILTERS
    # -------------------------
    def _is_low_value_question(q: str) -> bool:
        t = (q or "").strip().lower()

        banned_patterns = [
            r"\bwhich\b.*\bnot\b",
            r"\bnot listed\b",
            r"\bnot stated\b",
            r"\bexcept\b",
            r"\ball of the following\b.*\bexcept\b",
            r"\bnone of the above\b",
            r"\breference\b",
            r"\breferences\b",
            r"\burl\b",
            r"\blink\b",
            r"\byoutube\b",
            r"\bpage\b\s*\d+",
            r"\bwhat is the purpose\b",
            r"\bpurpose of\b.*\bmaterial\b",
            r"\bwhat does this reading\b",
        ]
        for pat in banned_patterns:
            if re.search(pat, t):
                return True

        # ultra short -> usually low insight
        if len(t) < 35:
            return True

        return False

    def _explanation_has_reasoning(exp: str) -> bool:
        e = (exp or "").strip()
        if len(e) < 40:
            return False
        if "reason" in e.lower():
            return True
        # at least 2 sentences heuristic
        if len(re.split(r"(?<=[.!?])\s+", e)) >= 2:
            return True
        return False

    # -------------------------
    # DUPLICATE / REPHRASE GUARDS
    # -------------------------
    used_context_keys: set[str] = set()
    topic_counts: Dict[str, int] = {}
    seen_questions_norm: List[str] = []

    def _context_key(meta: Dict[str, Any]) -> str:
        f = (meta or {}).get("file") or ""
        p = (meta or {}).get("page")
        s = (meta or {}).get("slide")
        if p is not None:
            return f"pdf::{f}::p{p}"
        if s is not None:
            return f"pptx::{f}::s{s}"
        return f"file::{f}"

    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        stop = {"the", "and", "or", "to", "of", "in", "a", "an", "is", "are", "was", "were", "be", "by", "for", "with"}
        return {w for w in words if len(w) > 2 and w not in stop}

    def _too_similar(new_q: str) -> bool:
        n = _norm_q(new_q)
        if not n:
            return True

        new_tok = _tokens(n)
        if not new_tok:
            return True

        for old in seen_questions_norm:
            old_tok = _tokens(old)
            inter = len(new_tok & old_tok)
            union = len(new_tok | old_tok)
            jacc = (inter / union) if union else 0.0
            seq = SequenceMatcher(None, n, old).ratio()

            # tuned to block rephrases
            if jacc >= 0.55 or seq >= 0.82:
                return True
        return False

    def _topic_cap(topic: str) -> int:
        return max(2, (n_questions + 2) // 4)

    # -------------------------
    # 1) BROAD SAMPLE for BLUEPRINT
    # -------------------------
    _, broad_docs, success = get_relevant_rag(
        db_file="chat.db",
        rag_index_dir=subject_dir,
        target_id=subject,
        query=f"{subject} key concepts definitions formulas examples common mistakes",
        model_name=model_name,
        top_k=30,
        vector_db=vector_db,
        suffix=suffix,
    )
    if not success or not broad_docs:
        return []

    sample_notes = "\n\n".join((d.page_content or "") for d in broad_docs[:12])
    sample_notes = (sample_notes or "")[:12000]

    blueprint = _llm_json(model_name, _blueprint_system(), _blueprint_user(subject, sample_notes))
    topics = (blueprint or {}).get("topics", []) or []
    if not topics:
        topics = [{"topic": subject, "question_styles": ["application"], "keywords": [subject]}]

    topic_pool = []
    for t in topics[:12]:
        topic = str(t.get("topic", "")).strip() or subject
        styles = t.get("question_styles") or ["application"]
        keywords = t.get("keywords") or [topic]
        kw = [k for k in keywords if isinstance(k, str) and k.strip()]
        topic_pool.append((topic, styles, kw))

    call_fn = _call_openai_for_chunk if is_openai else _call_ollama_for_chunk

    # -------------------------
    # 2) GENERATION LOOP
    # -------------------------
    final_quiz: List[Dict[str, Any]] = []

    attempts = 0
    max_attempts = max(50, n_questions * 12)

    while len(final_quiz) < n_questions and attempts < max_attempts:
        attempts += 1

        # bias toward underused topics
        random.shuffle(topic_pool)
        topic, styles, keywords = min(topic_pool, key=lambda t: topic_counts.get(t[0], 0))

        # respect topic cap
        if topic_counts.get(topic, 0) >= _topic_cap(topic):
            alternatives = [t for t in topic_pool if topic_counts.get(t[0], 0) < _topic_cap(t[0])]
            if alternatives:
                topic, styles, keywords = random.choice(alternatives)

        style = random.choice(styles) if styles else "application"

        # per-topic queries (push variety + calculations/graphs)
        queries = []
        if keywords:
            for k in keywords[:3]:
                queries.append(f"{topic} {k} explanation example")
                queries.append(f"{topic} {k} common misconception")
        else:
            queries.append(f"{topic} explanation example")
            queries.append(f"{topic} common misconception")

        # extra variety drivers
        queries.append(f"{topic} calculation worked example")
        queries.append(f"{topic} graph table interpretation")
        queries.append(f"{topic} application scenario best method")

        # retrieve docs
        per_topic_docs = []
        seen_meta = set()

        for q in queries:
            _, docs, ok = get_relevant_rag(
                db_file="chat.db",
                rag_index_dir=subject_dir,
                target_id=subject,
                query=q,
                model_name=model_name,
                top_k=12,
                vector_db=vector_db,
                suffix=suffix,
            )
            if not ok or not docs:
                continue

            for d in docs:
                meta = getattr(d, "metadata", {}) or {}
                key = (meta.get("file"), meta.get("page"), meta.get("slide"), (d.page_content or "")[:120])
                if key in seen_meta:
                    continue
                seen_meta.add(key)
                per_topic_docs.append(d)

        if not per_topic_docs:
            continue

        random.shuffle(per_topic_docs)

        # choose doc with NEW context to avoid repeated "Show Relevant Context"
        doc = None
        for cand in per_topic_docs:
            meta = getattr(cand, "metadata", {}) or {}
            ck = _context_key(meta)
            if ck not in used_context_keys:
                doc = cand
                break
        if doc is None:
            continue

        chunk = (doc.page_content or "").strip()
        if len(chunk) < 140:
            continue

        # generate ONE question
        qs = call_fn(model_name, chunk, subject, max_q=1, context_meta=None)
        if not qs:
            continue

        qobj = qs[0]
        qobj["context_meta"] = getattr(doc, "metadata", {}) or {}

        ck = _context_key(qobj["context_meta"])
        if ck in used_context_keys:
            continue

        # hard reject low-value / rephrased duplicates
        if _is_low_value_question(qobj.get("question", "")):
            continue
        if _too_similar(qobj.get("question", "")):
            continue

        # verify + fix
        mcq_for_verify = {
            "question": qobj.get("question", ""),
            "choices": qobj.get("choices", []),
            "answer_index": int(qobj.get("answer_idx", 0)),
            "explanation": qobj.get("explanation", ""),
            "difficulty": qobj.get("difficulty", "medium"),
            "topic": qobj.get("topic", topic) or topic,
        }

        v = _llm_json(model_name, _verify_system(), _verify_user(chunk, mcq_for_verify))
        if not v:
            continue

        ok = bool(v.get("ok", False))
        fixed = v.get("fixed") if isinstance(v.get("fixed"), dict) else None
        if not ok and not fixed:
            continue

        item = fixed if fixed else mcq_for_verify

        # validate output
        choices = item.get("choices", []) or []
        if not isinstance(choices, list) or len(choices) != 4:
            continue

        try:
            ans = int(item.get("answer_index", 0))
        except Exception:
            continue
        if ans < 0 or ans > 3:
            continue

        question_text = str(item.get("question", "")).strip()
        explanation_text = str(item.get("explanation", "")).strip()

        if _is_low_value_question(question_text):
            continue
        if _too_similar(question_text):
            continue
        if not _explanation_has_reasoning(explanation_text):
            continue

        final_topic = str(item.get("topic", topic)).strip() or topic

        upgraded = {
            "question": question_text,
            "choices": [str(c).strip() for c in choices],
            "answer_idx": ans,
            "explanation": explanation_text,
            "needs_image": False,
            "image_hint": "",
            "source_hint": " ".join(chunk.split()[:30]),
            "difficulty": str(item.get("difficulty", "medium")).strip() or "medium",
            "topic": final_topic,
            "context_meta": qobj["context_meta"],
        }

        # commit trackers
        used_context_keys.add(ck)
        seen_questions_norm.append(_norm_q(upgraded["question"]))
        topic_counts[final_topic] = topic_counts.get(final_topic, 0) + 1
        final_quiz.append(upgraded)

    return final_quiz

# modules/quizstats/quiz_engine.py
from __future__ import annotations

import json
import os
import random
import re
import ollama
from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher
from openai import OpenAI

# NEW: use the subject index so we know which slide/page a question came from
from modules.quizstats.file_utils import get_subject_index_entries
from modules.functions import get_openai_client, get_relevant_rag


# -------------------------------------------------------------
#   JSON + TEXT HELPERS
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


def _strip_choice_label(s: str) -> str:
    return re.sub(r"^[ABCD]\s*[\)\.\-:]\s*", "", (s or "").strip(), flags=re.IGNORECASE)


def _normalize_choices_keep_text_only(choices: List[str]) -> List[str]:
    """
    Returns 4 choices WITHOUT forcing A)/B)/C)/D) prefixes.
    We'll add labels later after balancing, so labels don't bias anything.
    """
    clean = [str(c).strip() for c in (choices or []) if str(c).strip()]
    if len(clean) < 4:
        return []
    clean = [_strip_choice_label(c) for c in clean]
    return clean[:4]


def _apply_choice_labels(choices: List[str]) -> List[str]:
    """
    Adds 'A) '...'D) ' and ensures any math renders properly:
    - Keep label OUTSIDE math
    - Wrap math-looking expressions in $...$ if not already
    """
    labels = ["A) ", "B) ", "C) ", "D) "]
    out = []
    for lab, c in zip(labels, choices):
        c2 = _strip_choice_label(c)
        c2 = _normalize_math_text(c2)
        c2 = _ensure_math_wrapped_if_needed(c2)
        out.append(f"{lab}{c2}")
    return out


# -------------------------------------------------------------
#   MATH RENDERING FIXES (THIS IS THE MAIN CHANGE)
# -------------------------------------------------------------
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

    # Already math-wrapped somewhere
    if "$" in t:
        return True

    # Strong LaTeX signals
    for trig in _LATEX_TRIGGERS:
        if trig in t:
            return True

    # Common math tokens (keep conservative)
    if re.search(r"[\^_]", t):  # exponents/subscripts
        return True
    if re.search(r"[=<>]", t):
        return True
    if re.search(r"\b(s|t|x|y|r)\b", t) and re.search(r"\d", t):
        return True

    return False


def _ensure_math_wrapped_if_needed(text: str) -> str:
    """
    Streamlit renders LaTeX only inside $...$ (in markdown/radio labels).
    If the text is mathy and not already wrapped, wrap the WHOLE expression.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # If there is any $ already, don't force-wrap (avoid breaking mixed text).
    if "$" in t:
        return t

    if _looks_like_math(t):
        return f"${t}$"

    return t


def _normalize_math_text(text: str) -> str:
    """
    Make math display-friendly for Streamlit:
    - Collapse double-escaped backslashes (\\frac -> \frac)
    - Convert \( \) and \[ \] to $...$
    - Convert common plain-text fractions to \frac{...}{...} where safe
    - Leave final wrapping to _ensure_math_wrapped_if_needed()
    """
    if not text:
        return ""

    out = str(text)

    # If the LLM output contains double-escaped backslashes literally, fix it.
    # Example: "\\frac{1}{s}" should become "\frac{1}{s}" in the displayed string.
    out = out.replace("\\\\", "\\")

    # Normalize \(...\) and \[...\] to $...$ for Streamlit markdown.
    out = re.sub(r"\\\((.*?)\\\)", r"$\1$", out, flags=re.DOTALL)
    out = re.sub(r"\\\[(.*?)\\\]", r"$\1$", out, flags=re.DOTALL)

    # Remove unnecessary parentheses around equations like "( y = ... )".
    out = re.sub(r"\(\s*([A-Za-z][^()=]*=\s*[^()]+)\s*\)", r"\1", out)

    # Replace common "1/(...)" style with \frac{1}{...} (best-effort, conservative)
    # Only do this when it looks like a single fraction expression.
    if "/" in out and "\\" not in out and "$" not in out:
        # Simple forms like: 1/(s^2+1) or (s-1)/(s(s^2+1))
        # Avoid converting long sentences.
        if len(out.strip()) <= 80 and re.search(r"[A-Za-z0-9\)\]]\s*/\s*[\(\[]", out):
            out = re.sub(
                r"^\s*\(?\s*([A-Za-z0-9\^\_\+\-\s\(\)]+)\s*\)?\s*/\s*\(?\s*([A-Za-z0-9\^\_\+\-\s\(\)]+)\s*\)?\s*$",
                r"\\frac{\1}{\2}",
                out.strip(),
            )

    # Keep your earlier helpful replacements (don’t overdo it)
    replacements = {
        r"\bL\{([^}]+)\}": r"\\mathcal{L}\{\1\}",
        r"\bLaplace\s*\{\s*([^}]+)\s*\}": r"\\mathcal{L}\{\1\}",
        r"\bsqrt\(([^)]+)\)": r"\\sqrt{\1}",
    }
    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out)

    return out.strip()


# -------------------------------------------------------------
#   ANSWER INDEX PARSING (REMOVE "A BIAS")
# -------------------------------------------------------------
def _parse_answer_index(val: Any) -> int | None:
    """
    Parse answer_index robustly.
    Accepts:
      - 0..3
      - "0".."3"
      - "A"/"B"/"C"/"D" (and variants like "A)", "B.", "C )")
    Returns 0..3, or None if invalid.
    """
    if val is None:
        return None

    if isinstance(val, int):
        return val if 0 <= val <= 3 else None

    s = str(val).strip().upper()
    if not s:
        return None

    m = re.match(r"^([ABCD])", s)
    if m:
        return {"A": 0, "B": 1, "C": 2, "D": 3}[m.group(1)]

    try:
        n = int(s)
        return n if 0 <= n <= 3 else None
    except Exception:
        return None


# -------------------------------------------------------------
#   LLM JSON CALL
# -------------------------------------------------------------
def _llm_json(model_name: str, system: str, user: str) -> dict | None:
    is_openai_model = "gpt" in (model_name or "").lower()

    if is_openai_model:
        client = get_openai_client()
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
            options={"temperature": 0.2, "num_predict": 900},
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
#   CHOICE DIVERSITY FIX (REMOVE "A BIAS" + DUPLICATE HAZARDS)
# -------------------------------------------------------------
def _rebalance_answer_positions(quiz: List[Dict[str, Any]], seed: int | None = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    counts = [0, 0, 0, 0]

    for q in quiz:
        choices = list(q.get("choices") or [])
        ans = q.get("answer_idx", None)
        if not isinstance(choices, list) or len(choices) != 4:
            continue
        if not isinstance(ans, int) or not (0 <= ans <= 3):
            continue

        correct = choices[ans]
        distractors = [c for i, c in enumerate(choices) if i != ans]
        rng.shuffle(distractors)

        min_count = min(counts)
        candidates = [i for i, c in enumerate(counts) if c == min_count]
        target = rng.choice(candidates)

        new_choices = [None] * 4
        new_choices[target] = correct

        di = 0
        for i in range(4):
            if new_choices[i] is None:
                new_choices[i] = distractors[di]
                di += 1

        q["choices"] = new_choices
        q["answer_idx"] = target
        counts[target] += 1

    return quiz


def _has_duplicate_choices(choices: List[str]) -> bool:
    plain = [_strip_choice_label(c) for c in (choices or [])]
    norm = [re.sub(r"\s+", " ", p.lower()).strip() for p in plain if p.strip()]
    return len(norm) != 4 or len(set(norm)) != 4


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
#   PROMPTS
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
        "3) Choices MUST be 4 distinct answers (labels optional; we will label later).\n"
        "4) Ensure answer_index is correct (0=A, 1=B, 2=C, 3=D) OR use A/B/C/D.\n"
        "5) Explanation MUST include:\n"
        "   - a short direct quote from the notes AND\n"
        "   - 1–2 sentences of reasoning/inference/calculation (not just restating)\n"
        "6) Use LaTeX for math and KEEP it standard (e.g., \\frac{a}{b}).\n"
        "7) If the content is mathematical, include LaTeX in BOTH the question and the choices.\n"
        "8) Randomize which option is correct across questions; do NOT make A correct more often.\n"
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
- For math topics: use proper LaTeX like \\frac{{a}}{{b}}, \\sin(t), e^{{-t}}, etc.

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
#   OLLAMA / OPENAI GENERATORS
# -------------------------------------------------------------
def _call_ollama_for_chunk(model_name, chunk_text, subject, max_q, context_meta=None):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": _get_unified_system_msg()},
                {"role": "user", "content": _get_unified_user_prompt(max_q, subject, chunk_text)},
            ],
            options={"temperature": 0.1, "num_predict": 1400},
            format="json",
        )
        raw = response["message"]["content"]
    except Exception as e:
        print(f"[quiz_engine] Ollama error ({model_name}):", e)
        return []
    return _parse_raw_quiz_json(raw, chunk_text, context_meta)


def _call_openai_for_chunk(model_name, chunk_text, subject, max_q, context_meta=None):
    client = get_openai_client()
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
#   JSON PARSING
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
        choices = _normalize_choices_keep_text_only(q.get("choices", []))
        if len(choices) != 4 or not question:
            continue

        if _has_duplicate_choices(choices):
            continue

        ans = _parse_answer_index(q.get("answer_index", None))
        if ans is None:
            continue  # no default-to-A

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

        # Normalize and fix math display (wrap if needed happens later for choices via labeler;
        # for question, we also ensure math is readable)
        q_text = _normalize_math_text(question)
        q_text = q_text  # keep question as normal text; it may contain $...$ already

        cleaned_choices = []
        for c in choices:
            c2 = _normalize_math_text(c)
            # store label-free; we will wrap if needed when applying labels
            cleaned_choices.append(c2)

        out.append(
            {
                "question": q_text,
                "choices": cleaned_choices,  # label-free for now
                "answer_idx": ans,
                "explanation": _normalize_math_text(explanation),
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
        ans_word = words[idx]
        words[idx] = "_____"
        stem = " ".join(words)

        wrong_pool = [w for w in words if w != "_____"]
        if len(wrong_pool) < 3:
            continue
        wrongs = random.sample(wrong_pool, 3)

        choices_raw = [ans_word] + wrongs
        random.shuffle(choices_raw)

        choices = choices_raw[:4]
        if _has_duplicate_choices(choices):
            continue

        answer_idx = choices.index(ans_word) if ans_word in choices else 0

        out.append(
            {
                "question": _normalize_math_text(stem),
                "choices": [_normalize_math_text(c) for c in choices],
                "answer_idx": answer_idx,
                "explanation": _normalize_math_text("Look at the original sentence."),
                "needs_image": False,
                "image_hint": "",
                "source_hint": s[:40],
                "difficulty": "easy",
                "topic": "General",
                "context_meta": {},
            }
        )

    out = _rebalance_answer_positions(out, seed=None)
    for q in out:
        if isinstance(q.get("choices"), list) and len(q["choices"]) == 4:
            q["choices"] = _apply_choice_labels(q["choices"])
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
                max_q=min(4, remain),
                context_meta=ctx,
            )

            for q in qs:
                if all(q["question"] != e["question"] for e in quiz):
                    quiz.append(q)
                    if len(quiz) >= n_questions:
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
                max_q=min(4, remain),
                context_meta=None,
            )

            for q in qs:
                if all(q["question"] != e["question"] for e in quiz):
                    quiz.append(q)
                    if len(quiz) >= n_questions:
                        break

    if not quiz:
        return _fallback_quiz(corpus_text, subject, n_questions)

    quiz = _rebalance_answer_positions(quiz, seed=None)
    for q in quiz:
        if isinstance(q.get("choices"), list) and len(q["choices"]) == 4:
            q["choices"] = _apply_choice_labels(q["choices"])

        # Optional: make mathy questions slightly nicer if they are pure math:
        q["question"] = _normalize_math_text(q.get("question", ""))
        # (We do NOT wrap whole question in $...$ because many questions are mixed English + math)

    return quiz


# -------------------------------------------------------------
#   MAIN PUBLIC API (RAG)
# -------------------------------------------------------------
def build_quiz_from_rag(
    subject,
    model_name,
    n_questions: int = 10,
    vector_db: str = "faiss",
    user_id: int = None,
    ollama_models: tuple = (),
    openai_models: tuple = (),
    qdrant_url=None,
    qdrant_api_key=None
) -> List[Dict[str, Any]]:

    subject_dir = os.path.join("modules", "subjects", f"user_{user_id}", subject)
    is_openai = "gpt" in model_name.lower()
    suffix = "openai" if is_openai else "ollama"
    verify_enabled = False

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
        return len(t) < 28

    def _explanation_has_reasoning(exp: str) -> bool:
        e = (exp or "").strip()
        if len(e) < 30:
            return False
        if any(token in e.lower() for token in ("reason", "because", "therefore", "thus")):
            return True
        return len(re.split(r"(?<=[.!?])\s+", e)) >= 2

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
            if jacc >= 0.65 or seq >= 0.88:
                return True
        return False

    def _topic_cap(topic: str) -> int:
        return max(2, (n_questions + 2) // 4)

    _rag_cache: Dict[str, List[Any]] = {}

    def _cached_get_docs(query: str, top_k: int) -> List[Any]:
        key = f"{query}::k{top_k}"
        if key in _rag_cache:
            return _rag_cache[key]

        _, docs, ok = get_relevant_rag(
            db_file="chat.db",
            subject_dir=subject_dir,
            target_id=subject,
            query=query,
            model_name=model_name,
            ollama_models=ollama_models,
            openai_models=openai_models,
            top_k=top_k,
            vector_db=vector_db,
            suffix=suffix,
            user_id=user_id,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        res = docs if (ok and docs) else []
        _rag_cache[key] = res
        return res

    broad_docs = _cached_get_docs(
        query=f"{subject} key concepts definitions formulas examples common mistakes",
        top_k=30,
    )
    if not broad_docs:
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

    final_quiz: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max(25, n_questions * 3)

    while len(final_quiz) < n_questions and attempts < max_attempts:
        attempts += 1

        random.shuffle(topic_pool)
        topic, styles, keywords = min(topic_pool, key=lambda t: topic_counts.get(t[0], 0))

        if topic_counts.get(topic, 0) >= _topic_cap(topic):
            alternatives = [t for t in topic_pool if topic_counts.get(t[0], 0) < _topic_cap(t[0])]
            if alternatives:
                topic, styles, keywords = random.choice(alternatives)

        styles = styles or ["application"]
        style = random.choice(styles)

        kw_txt = " ".join(keywords[:2]) if keywords else ""
        query_txt = f"{topic} {kw_txt} {style} worked example calculation explanation"
        per_topic_docs = _cached_get_docs(query_txt, top_k=10)
        if not per_topic_docs:
            continue

        random.shuffle(per_topic_docs)

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

        qs = call_fn(model_name, chunk, subject, max_q=4, context_meta=None)
        if not qs:
            continue

        for qobj in qs:
            qobj["context_meta"] = getattr(doc, "metadata", {}) or {}

            ck = _context_key(qobj["context_meta"])
            if ck in used_context_keys:
                continue

            if _is_low_value_question(qobj.get("question", "")):
                continue
            if _too_similar(qobj.get("question", "")):
                continue

            mcq_for_verify = {
                "question": qobj.get("question", ""),
                "choices": qobj.get("choices", []),
                "answer_index": int(qobj.get("answer_idx", 0)),
                "explanation": qobj.get("explanation", ""),
                "difficulty": qobj.get("difficulty", "medium"),
                "topic": qobj.get("topic", topic) or topic,
            }

            item = mcq_for_verify
            if verify_enabled:
                v = _llm_json(model_name, _verify_system(), _verify_user(chunk, mcq_for_verify))
                if v:
                    ok = bool(v.get("ok", False))
                    fixed = v.get("fixed") if isinstance(v.get("fixed"), dict) else None
                    if not ok and not fixed:
                        continue
                    item = fixed if fixed else mcq_for_verify

            choices = _normalize_choices_keep_text_only(item.get("choices", []) or [])
            if not isinstance(choices, list) or len(choices) != 4:
                continue
            if _has_duplicate_choices(choices):
                continue

            ans = _parse_answer_index(item.get("answer_index", None))
            if ans is None:
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
                "question": _normalize_math_text(question_text),
                "choices": [_normalize_math_text(c) for c in choices],  # label-free for now
                "answer_idx": ans,
                "explanation": _normalize_math_text(explanation_text),
                "needs_image": False,
                "image_hint": "",
                "source_hint": " ".join(chunk.split()[:30]),
                "difficulty": str(item.get("difficulty", "medium")).strip() or "medium",
                "topic": final_topic,
                "context_meta": qobj["context_meta"],
            }

            used_context_keys.add(ck)
            seen_questions_norm.append(_norm_q(upgraded["question"]))
            topic_counts[final_topic] = topic_counts.get(final_topic, 0) + 1
            final_quiz.append(upgraded)

            if len(final_quiz) >= n_questions:
                break

    final_quiz = _rebalance_answer_positions(final_quiz, seed=None)
    for q in final_quiz:
        if isinstance(q.get("choices"), list) and len(q["choices"]) == 4:
            q["choices"] = _apply_choice_labels(q["choices"])
        q["question"] = _normalize_math_text(q.get("question", ""))

    return final_quiz

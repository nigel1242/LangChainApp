# modules/quiz_engine.py
import json
import random
from typing import Dict, Any

PROMPT = """
You are a tutor. Given the context below from the student's PDFs, create ONE multiple-choice question.
- Provide exactly 4 choices (A-D), one correct.
- Include a short explanation.
- Identify a concise topic and a difficulty (easy/medium/hard).
- Return STRICT JSON with keys: question, choices, answer_idx, explanation, topic, difficulty.

Context:
{context}

Return JSON ONLY.
"""

def _safe_parse_json(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(txt[start:end+1])
        raise

def generate_one_question(llm, vectorstore) -> Dict[str, Any]:
    # Retrieve context chunks
    docs = vectorstore.similarity_search("key concepts", k=4)
    context = "\n\n".join(d.page_content for d in docs)

    # Ask the LLM to generate a question
    resp = llm.predict(PROMPT.format(context=context))
    data = _safe_parse_json(resp)

    # Sanitize / Fill defaults
    data["choices"] = list(data.get("choices", []))[:4]
    while len(data["choices"]) < 4:
        data["choices"].append(f"Option {len(data['choices'])+1}")

    if "answer_idx" not in data or not isinstance(data["answer_idx"], int):
        data["answer_idx"] = random.randint(0, 3)

    data["topic"] = data.get("topic", "General")
    data["difficulty"] = data.get("difficulty", "easy")

    # ✅ FIXED: correct variable in comprehension
    data["source_ids"] = [
        getattr(doc, "metadata", {}).get("source", f"doc_{i}")
        for i, doc in enumerate(docs)
    ]

    # Return standardized dict
    return {
        "question": data.get("question", "No question generated."),
        "choices": data["choices"],
        "answer_idx": int(data["answer_idx"]),
        "explanation": data.get("explanation", "No explanation provided."),
        "topic": data["topic"],
        "difficulty": data["difficulty"],
        "source_ids": data["source_ids"],
    }

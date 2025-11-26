import streamlit as st
from dotenv import load_dotenv

# Optional HTML templates
try:
    from htmlTemplates import css, bot_template, user_template
    _USE_HTML = True
except Exception:
    _USE_HTML = False
    css = bot_template = user_template = ""

# Import local modules (module object, not just symbols)
import modules.llm_stack as llm_stack
from modules.rag_index import pdf_to_docs, build_faiss
from modules.chat_chain import make_chat_chain
from modules.quiz_engine import generate_one_question
from modules.quiz_store import init_db, record_result, fetch_stats, backend_name  # TinyDB backend

# ---- App Setup ----
load_dotenv()
st.set_page_config(page_title="My Learning AI", page_icon="📚", layout="wide")
if _USE_HTML and css:
    st.write(css, unsafe_allow_html=True)

# ---- Session defaults ----
defaults = {
    "vectorstore": None,      # FAISS store
    "chat": None,             # ConversationalRetrievalChain
    "chat_history": [],
    "current_q": None,        # current quiz question dict
    "start_ms": 0,            # question start timestamp (ms)
    "db_ready": False,        # TinyDB initialised?
    "db_path": "quiz_results.json",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Clean any leftover SQLite objects from old versions
for old_key in ("db", "db_conn", "sqlite_conn"):
    if old_key in st.session_state:
        try:
            if hasattr(st.session_state[old_key], "close"):
                st.session_state[old_key].close()
        except Exception:
            pass
        del st.session_state[old_key]

# ---------- Compat shim so old llm_stack still works ----------
def _safe_get_embeddings(use_local_flag: bool):
    """
    Try new signature get_embeddings(use_local=...), else fallback to legacy get_embeddings().
    This prevents 'unexpected keyword argument use_local' crashes if an old module is cached.
    """
    try:
        # Try calling with keyword (new API)
        return llm_stack.get_embeddings(use_local=use_local_flag)
    except TypeError:
        try:
            # Try positional (new API still accepts it)
            return llm_stack.get_embeddings(use_local_flag)
        except TypeError:
            # Final fallback to legacy no-arg function
            return llm_stack.get_embeddings()

def _llm():
    return llm_stack.get_llm()

# ---- Sidebar: Upload PDFs & Indexing ----
with st.sidebar:
    st.header("Your documents")
    pdfs = st.file_uploader(
        "Upload PDFs (or slides exported to PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.write("**Indexing options**")
    use_local_emb = st.checkbox("Use local MiniLM embeddings (faster)", value=True)
    c1, c2 = st.columns(2)
    page_from = c1.number_input("From page", min_value=1, value=1, step=1)
    page_to   = c2.number_input("To page (0 = all)", min_value=0, value=0, step=1)

    # Debug helpers (render which llm_stack was imported and function signature)
    with st.expander("Debug (imports)"):
        st.code(f"llm_stack file: {getattr(llm_stack, '__file__', 'unknown')}", language="text")
        try:
            st.code(f"get_embeddings signature: {inspect.signature(llm_stack.get_embeddings)}", language="text")
        except Exception as e:
            st.code(f"Could not inspect signature: {e}", language="text")

    if st.button("Process"):
        if not pdfs:
            st.warning("Please upload at least one PDF first.")
        else:
            progress_files = st.progress(0, text="Reading PDFs…")
            status = st.empty()

            def on_progress(done, total):
                pct = int(done / max(1, total) * 100)
                progress_files.progress(pct, text=f"Reading PDFs… ({done}/{total})")

            with st.spinner("Processing your PDFs..."):
                # Embeddings (compat-safe)
                emb = _safe_get_embeddings(use_local_emb)
                st.caption(f"Embeddings mode: **{'Local MiniLM' if use_local_emb else 'OpenAI'}**")

                # Extract & chunk quickly with PyMuPDF
                chunks = pdf_to_docs(
                    pdfs,
                    page_from=page_from,
                    page_to=(None if page_to == 0 else int(page_to)),
                    on_progress=on_progress
                )
                status.write(f"Chunked {len(chunks)} passages. Building index…")

                # Build FAISS & chat chain
                vs = build_faiss(chunks, emb)
                st.session_state.vectorstore = vs
                st.session_state.chat = make_chat_chain(_llm(), vs)

                status.write("Index built.")
                progress_files.empty()
                st.success("✅ Ready! You can now chat or take a quiz.")

# Helper: generate new quiz question
def _new_question():
    q = generate_one_question(_llm(), st.session_state.vectorstore)
    st.session_state.current_q = q
    st.session_state.start_ms = int(time.time() * 1000)

# ---- Main Tabs ----
st.title("My Learning AI")
st.caption(f"Results backend: **{backend_name()}**")  # should show "tinydb"
tabs = st.tabs(["💬 Chat", "📝 Quiz", "📊 Stats"])

# ---- 💬 CHAT TAB ----
with tabs[0]:
    st.subheader("Chat with your PDFs")
    query = st.text_input("Ask something about your uploaded documents:")
    if query and st.session_state.chat:
        response = st.session_state.chat({"question": query})
        st.session_state.chat_history = response.get("chat_history", [])
        for msg in st.session_state.chat_history:
            role = "You" if getattr(msg, "type", "ai") == "human" else "Tutor"
            content = msg.content
            if _USE_HTML and role == "Tutor" and bot_template:
                st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)
            elif _USE_HTML and role == "You" and user_template:
                st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
            else:
                st.markdown(f"**{role}:** {content}")
    elif not st.session_state.chat:
        st.info("Please upload and process PDFs first.")

# ---- 📝 QUIZ TAB ----
with tabs[1]:
    st.subheader("Auto-generated quiz")
    if not st.session_state.vectorstore:
        st.info("Upload and process your documents first.")
    else:
        if st.button("Generate a new question"):
            _new_question()

        q = st.session_state.get("current_q")
        if q:
            st.write(f"**Topic:** {q['topic']} | **Difficulty:** {q['difficulty']}")
            st.write(f"**Question:** {q['question']}")

            letters = ["A", "B", "C", "D"]
            raw_choices = q.get("choices", [])
            choices = [(str(c) if c else f"Option {i+1}") for i, c in enumerate(raw_choices)]
            while len(choices) < 4:
                choices.append(f"Option {len(choices)+1}")
            choices = choices[:4]

            labels = ["— Select an answer —"] + [f"{letters[i]}. {choices[i]}" for i in range(4)]
            selection = st.radio(
                "Choose one:",
                labels,
                index=0,
                key=f"choice_{hash(q['question'])}"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit"):
                    if selection == "— Select an answer —":
                        st.warning("Please select an answer first.")
                    else:
                        picked_idx = labels.index(selection) - 1
                        correct = (picked_idx == int(q["answer_idx"]))
                        st.success("✅ Correct!" if correct else "❌ Incorrect.")
                        st.write(f"**Explanation:** {q['explanation']}")
                        st.caption(f"Sources: {', '.join(q['source_ids'])}")

                        # Save result using TinyDB
                        if not st.session_state.db_ready:
                            st.session_state.db_path = init_db(st.session_state.get("db_path"))
                            st.session_state.db_ready = True

                        elapsed = int(time.time() * 1000) - st.session_state.start_ms
                        record_result(
                            st.session_state.db_path,
                            q["topic"], q["difficulty"], correct, elapsed, q["source_ids"]
                        )

            with col2:
                st.button("Next question", on_click=_new_question)
        else:
            st.info("Click **Generate a new question** to start.")

# ---- 📊 STATS TAB ----
with tabs[2]:
    st.subheader("Quiz statistics")
    if st.session_state.get("db_ready"):
        try:
            stats = fetch_stats(st.session_state.db_path)
            total, avg_ms = stats["overall"] if stats["overall"] else (0, 0.0)
            c1, c2 = st.columns(2)
            c1.metric("Total answered", int(total or 0))
            c2.metric("Avg time (s)", round((avg_ms or 0.0) / 1000.0, 2))

            st.markdown("### Accuracy by Topic")
            for topic, acc, n in stats["by_topic"]:
                st.write(f"- **{topic}** — {acc:.1f}% ({n} questions)")

            st.markdown("### Accuracy by Difficulty")
            for diff, acc, n in stats["by_diff"]:
                st.write(f"- **{diff}** — {acc:.1f}% ({n} questions)")
        except Exception as e:
            st.error(f"Could not load stats: {e}")
            st.caption("Try: Menu ▸ Clear cache, then rerun the app.")
    else:
        st.info("Answer at least one question to see your statistics.")

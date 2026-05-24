"""
ledger-mind/app/streamlit_app.py

Streamlit UI for the Ledger-Mind RAG system.

Place this file here:
ledger-mind/app/streamlit_app.py

Run from project root:
streamlit run app/streamlit_app.py

This UI is intentionally built as a clean frontend layer. It does not force you
to rewrite your existing RAG, fine-tuning, chunking, or evaluation code.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# =============================================================================
# 1. PROJECT PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
QA_PAIRS_PATH = PROCESSED_DIR / "qa_pairs.json"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
GOLDEN_TEST_PATH = PROCESSED_DIR / "golden_test_set.jsonl"


# =============================================================================
# 2. PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Ledger-Mind | RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 3. CUSTOM CSS
# =============================================================================

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #312e81 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.25);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        max-width: 900px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.15rem;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
    }

    .answer-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
    }

    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .source-meta {
        color: #475569;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .source-text {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 1rem;
        color: #9a3412;
    }

    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 1rem;
        color: #166534;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# 4. DATA LOADING HELPERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_chunks() -> List[Dict[str, Any]]:
    data = load_json(CHUNKS_PATH)
    return data if isinstance(data, list) else []


@st.cache_data(show_spinner=False)
def load_qa_pairs() -> List[Dict[str, Any]]:
    data = load_json(QA_PAIRS_PATH)
    return data if isinstance(data, list) else []


def truncate(text: str, max_chars: int = 700) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def file_badge(path: Path) -> str:
    return "✅ Available" if path.exists() else "❌ Missing"


# =============================================================================
# 5. FALLBACK LOCAL RETRIEVER
# =============================================================================

class SimpleKeywordRetriever:
    """
    Simple fallback retriever.

    This is not your final RAG engine. It exists so the UI works immediately
    before you connect Weaviate/vector search.
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        terms = [term.lower().strip() for term in query.split() if len(term.strip()) > 2]
        scored: List[Tuple[int, Dict[str, Any]]] = []

        for chunk in self.chunks:
            text = str(chunk.get("text", "")).lower()
            score = sum(text.count(term) for term in terms)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


# =============================================================================
# 6. RAG BACKEND ADAPTER
# =============================================================================

class RagBackend:
    """
    UI-facing backend wrapper.

    Later, connect this class to your real code, for example:

        from ledger_mind.rag.pipeline import RagPipeline
        self.pipeline = RagPipeline(config_path="configs/rag.yaml")
        result = self.pipeline.answer(query=query, top_k=top_k, mode=mode)

    Expected result format:
        {
            "answer": "final answer here",
            "sources": [chunk dictionaries]
        }
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.retriever = SimpleKeywordRetriever(chunks)

    def answer(self, query: str, top_k: int, mode: str) -> Dict[str, Any]:
        retrieved = self.retriever.retrieve(query=query, top_k=top_k)

        if not retrieved:
            return {
                "answer": (
                    "No relevant chunk was found from chunks.json. "
                    "Either the question is outside the report, or the real vector retriever is not connected yet."
                ),
                "sources": [],
                "backend_status": "fallback_no_result",
            }

        evidence_lines = []
        for index, chunk in enumerate(retrieved, start=1):
            chunk_id = chunk.get("chunk_id", "unknown_chunk")
            page_start = chunk.get("page_start", "?")
            text = truncate(chunk.get("text", ""), 450)
            evidence_lines.append(f"[{index}] {chunk_id} | page {page_start}\n{text}")

        answer_text = (
            f"Mode selected: {mode}\n\n"
            "This screen is currently using the local fallback keyword retriever. "
            "That means the UI is ready, but your final industry-level backend should still connect Weaviate + embeddings + LLM generation.\n\n"
            "Relevant evidence found:\n\n"
            + "\n\n".join(evidence_lines)
        )

        return {
            "answer": answer_text,
            "sources": retrieved,
            "backend_status": "fallback_keyword_retriever",
        }


# =============================================================================
# 7. LOAD DATA
# =============================================================================

chunks = load_chunks()
qa_pairs = load_qa_pairs()
train_rows = load_jsonl(TRAIN_PATH)
golden_rows = load_jsonl(GOLDEN_TEST_PATH)
backend = RagBackend(chunks=chunks)


# =============================================================================
# 8. SESSION STATE
# =============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =============================================================================
# 9. SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📊 Ledger-Mind")
    st.caption("RAG + Fine-tuning UI for annual report question answering")

    st.divider()

    st.subheader("Answer Settings")

    answer_mode = st.selectbox(
        "Pipeline mode",
        options=[
            "Direct RAG",
            "Fine-tuned Model",
            "Hybrid RAG + Fine-tuned Style",
        ],
        index=0,
    )

    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5)
    show_sources = st.toggle("Show sources", value=True)
    show_raw_chunks = st.toggle("Show raw chunk text", value=False)

    st.divider()

    st.subheader("Data Status")
    st.write(f"chunks.json: {file_badge(CHUNKS_PATH)}")
    st.write(f"qa_pairs.json: {file_badge(QA_PAIRS_PATH)}")
    st.write(f"train.jsonl: {file_badge(TRAIN_PATH)}")
    st.write(f"golden_test_set.jsonl: {file_badge(GOLDEN_TEST_PATH)}")

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.rerun()


# =============================================================================
# 10. HERO SECTION
# =============================================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Ledger-Mind</div>
        <div class="hero-subtitle">
            A source-grounded RAG and fine-tuning assistant for answering questions from Uber's annual report.
            Built for retrieval, evidence inspection, Q/A dataset review, and final demo presentation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 11. METRICS
# =============================================================================

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

with metric_col_1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Chunks</div>
            <div class="metric-value">{len(chunks)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col_2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Q/A Pairs</div>
            <div class="metric-value">{len(qa_pairs)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col_3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Training Rows</div>
            <div class="metric-value">{len(train_rows)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col_4:
    api_status = "Ready" if os.getenv("OPENAI_API_KEY") else "Missing"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">OpenAI API Key</div>
            <div class="metric-value">{api_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =============================================================================
# 12. MAIN TABS
# =============================================================================

tab_chat, tab_sources, tab_dataset, tab_eval, tab_system = st.tabs(
    ["💬 Ask", "📚 Sources", "🧪 Q/A Dataset", "📈 Evaluation", "⚙️ System"]
)


# -----------------------------------------------------------------------------
# TAB 1: ASK
# -----------------------------------------------------------------------------

with tab_chat:
    left_col, right_col = st.columns([0.67, 0.33], gap="large")

    with left_col:
        st.subheader("Ask from the annual report")

        examples = [
            "What are Uber's reportable segments?",
            "Summarize Uber's business model.",
            "What are the main risk factors mentioned by Uber?",
            "How does Uber generate revenue?",
            "Explain Uber's platform strategy in simple terms.",
        ]

        selected_example = st.selectbox(
            "Example question",
            options=["Write my own question"] + examples,
        )

        default_query = "" if selected_example == "Write my own question" else selected_example

        query = st.text_area(
            "Question",
            value=default_query,
            height=130,
            placeholder="Ask a question from the report...",
        )

        ask_button = st.button("Generate Answer", type="primary", use_container_width=True)

        if ask_button:
            if not query.strip():
                st.warning("Enter a question first.")
            elif not chunks:
                st.error("chunks.json is missing or empty. Complete the PDF chunking task first.")
            else:
                with st.spinner("Retrieving evidence and generating answer..."):
                    result = backend.answer(
                        query=query.strip(),
                        top_k=top_k,
                        mode=answer_mode,
                    )
                    st.session_state.last_result = result
                    st.session_state.chat_history.append(
                        {
                            "question": query.strip(),
                            "answer": result.get("answer", ""),
                            "mode": answer_mode,
                            "sources": result.get("sources", []),
                        }
                    )

        if st.session_state.last_result:
            result = st.session_state.last_result
            st.markdown("### Answer")
            st.markdown(
                f"""
                <div class="answer-box">
                    <pre style="white-space: pre-wrap; color: #0f172a; font-family: inherit; font-size: 0.98rem; line-height: 1.6;">{result.get('answer', '')}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if show_sources:
                st.markdown("### Retrieved Sources")
                sources = result.get("sources", [])
                if not sources:
                    st.info("No sources retrieved.")
                for index, source in enumerate(sources, start=1):
                    chunk_id = source.get("chunk_id", "unknown_chunk")
                    page_start = source.get("page_start", "?")
                    page_end = source.get("page_end", page_start)
                    text = source.get("text", "")

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-meta">Source {index} · {chunk_id} · pages {page_start}-{page_end}</div>
                            <div class="source-text">{truncate(text, 900)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with right_col:
        st.subheader("Chat History")

        if not st.session_state.chat_history:
            st.info("No questions asked yet.")
        else:
            for item in reversed(st.session_state.chat_history[-6:]):
                with st.expander(item["question"]):
                    st.caption(f"Mode: {item['mode']}")
                    st.write(truncate(item["answer"], 500))


# -----------------------------------------------------------------------------
# TAB 2: SOURCES
# -----------------------------------------------------------------------------

with tab_sources:
    st.subheader("Inspect processed chunks")

    if not chunks:
        st.error("No chunks loaded. Check data/processed/chunks.json.")
    else:
        search_term = st.text_input("Search inside chunks", placeholder="Example: revenue, mobility, delivery, risk")

        filtered_chunks = chunks
        if search_term.strip():
            term = search_term.lower().strip()
            filtered_chunks = [
                chunk for chunk in chunks
                if term in str(chunk.get("text", "")).lower()
                or term in str(chunk.get("chunk_id", "")).lower()
            ]

        st.caption(f"Showing {len(filtered_chunks)} of {len(chunks)} chunks")

        table_rows = []
        for chunk in filtered_chunks[:300]:
            table_rows.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "preview": truncate(chunk.get("text", ""), 160),
                }
            )

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=420)

        if show_raw_chunks:
            st.markdown("### Raw Chunk Preview")
            for chunk in filtered_chunks[:20]:
                with st.expander(str(chunk.get("chunk_id", "unknown_chunk"))):
                    st.write(chunk)


# -----------------------------------------------------------------------------
# TAB 3: DATASET
# -----------------------------------------------------------------------------

with tab_dataset:
    st.subheader("Synthetic Q/A Dataset Review")

    if not qa_pairs:
        st.warning("qa_pairs.json is missing or empty.")
    else:
        category_options = sorted(
            list({str(row.get("category", "Unknown")) for row in qa_pairs})
        )

        selected_category = st.selectbox(
            "Filter by category",
            options=["All"] + category_options,
        )

        filtered_qa = qa_pairs
        if selected_category != "All":
            filtered_qa = [
                row for row in qa_pairs
                if str(row.get("category", "Unknown")) == selected_category
            ]

        qa_table = []
        for row in filtered_qa[:500]:
            qa_table.append(
                {
                    "category": row.get("category"),
                    "question": row.get("question") or row.get("input"),
                    "answer": truncate(row.get("answer") or row.get("output"), 180),
                    "source_chunk_id": row.get("source_chunk_id"),
                    "page_reference": row.get("page_reference"),
                }
            )

        st.dataframe(pd.DataFrame(qa_table), use_container_width=True, height=450)

    st.divider()
    st.subheader("Fine-tuning Training Rows")

    if not train_rows:
        st.warning("train.jsonl is missing or empty.")
    else:
        train_table = []
        for row in train_rows[:300]:
            train_table.append(
                {
                    "instruction": truncate(row.get("instruction", ""), 100),
                    "input": truncate(row.get("input", ""), 120),
                    "output": truncate(row.get("output", ""), 180),
                    "category": row.get("category"),
                    "source_chunk_id": row.get("source_chunk_id"),
                }
            )
        st.dataframe(pd.DataFrame(train_table), use_container_width=True, height=420)


# -----------------------------------------------------------------------------
# TAB 4: EVALUATION
# -----------------------------------------------------------------------------

with tab_eval:
    st.subheader("Evaluation Dashboard")

    st.markdown(
        """
        <div class="warning-box">
            This tab is a placeholder for your final evaluation layer. After Task 6, connect this page to your metrics file,
            for example: retrieval recall@k, answer faithfulness, ROUGE/BERTScore, latency, and cost per answer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    eval_col_1, eval_col_2, eval_col_3 = st.columns(3)
    with eval_col_1:
        st.metric("Retrieval Recall@5", "Pending")
    with eval_col_2:
        st.metric("Answer Faithfulness", "Pending")
    with eval_col_3:
        st.metric("Avg Latency", "Pending")

    st.divider()

    if golden_rows:
        st.subheader("Golden Test Set")
        golden_table = []
        for row in golden_rows[:300]:
            golden_table.append(
                {
                    "instruction": truncate(row.get("instruction", ""), 120),
                    "input": truncate(row.get("input", ""), 160),
                    "expected_output": truncate(row.get("output", ""), 220),
                    "source_chunk_id": row.get("source_chunk_id"),
                }
            )
        st.dataframe(pd.DataFrame(golden_table), use_container_width=True, height=430)
    else:
        st.info("golden_test_set.jsonl not found yet.")


# -----------------------------------------------------------------------------
# TAB 5: SYSTEM
# -----------------------------------------------------------------------------

with tab_system:
    st.subheader("System Overview")

    st.markdown(
        """
        <div class="success-box">
            Recommended architecture: PDF → cleaned chunks → embeddings/vector DB → retriever → LLM answer generator → source-grounded UI.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    system_rows = [
        {"Layer": "Data", "Current file": "data/raw/uber_2024_annual_report.pdf", "Status": "Input PDF"},
        {"Layer": "Chunking", "Current file": "data/processed/chunks.json", "Status": file_badge(CHUNKS_PATH)},
        {"Layer": "Synthetic Q/A", "Current file": "data/processed/qa_pairs.json", "Status": file_badge(QA_PAIRS_PATH)},
        {"Layer": "Fine-tuning", "Current file": "data/processed/train.jsonl", "Status": file_badge(TRAIN_PATH)},
        {"Layer": "Evaluation", "Current file": "data/processed/golden_test_set.jsonl", "Status": file_badge(GOLDEN_TEST_PATH)},
        {"Layer": "UI", "Current file": "app/streamlit_app.py", "Status": "✅ Current file"},
    ]

    st.dataframe(pd.DataFrame(system_rows), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("What to connect next")
    st.markdown(
        """
        1. Replace `SimpleKeywordRetriever` with your Weaviate retriever.  
        2. Replace the fallback answer in `RagBackend.answer()` with your OpenAI/Gemini/Claude answer generator.  
        3. Save evaluation metrics into `artifacts/reports/evaluation_metrics.json`.  
        4. Read that metrics file inside the Evaluation tab.  
        5. Keep this UI as the final demo interface for your assignment.
        """
    )

    with st.expander("Environment"):
        st.write("Project root:", str(PROJECT_ROOT))
        st.write("Python source path:", str(SRC_PATH))
        st.write("OpenAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))

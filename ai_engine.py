# ai_engine.py
import json
import time
from typing import List, Dict, Tuple

import numpy as np
import openai
import streamlit as st

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"   # or "gpt-4o-mini"

# 1-hour session per browser window
SESSION_TTL_SECONDS = 60 * 60

openai.api_key = st.secrets["openai"]["api_key"]


# -------------------------------------------------------------------
# Vector stores: ops (forecasts/rosters/docs) + customer (reviews/etc)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vector_store() -> Dict[str, Dict]:
    """
    Load both vector indices once per process.

    ai_index/vectors.npy + meta.json        -> operational snippets
    ai_index/mecca_vectors.npy + mecca_meta.json -> reviews, articles, competitors
    """
    def _normalise(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        return v / norms

    # --- Ops index ---
    ops_vectors = np.load("ai_index/vectors.npy").astype("float32")
    with open("ai_index/meta.json", "r", encoding="utf-8") as f:
        ops_meta = json.load(f)

    # --- Customer / market index ---
    cust_vectors = np.load("ai_index/mecca_vectors.npy").astype("float32")
    # 👇 this was causing your UnicodeDecodeError, now forced to utf-8
    with open("ai_index/mecca_meta.json", "r", encoding="utf-8") as f:
        cust_meta = json.load(f)

    store = {
        "ops": {
            "vectors": _normalise(ops_vectors),
            "meta": ops_meta,
        },
        "customer": {
            "vectors": _normalise(cust_vectors),
            "meta": cust_meta,
        },
    }
    return store


def embed_query(text: str) -> np.ndarray:
    resp = openai.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-8)


def search_index(query: str, k: int = 8, mode: str = "blended") -> List[Dict]:
    """
    mode:
        - "ops_only"       -> only forecast/roster/docs
        - "customer_only"  -> only reviews/news/competitors
        - "blended"/other  -> mix of both, sorted by similarity
    """
    store = load_vector_store()
    q = embed_query(query)

    results: List[Dict] = []

    def _score_index(name: str):
        idx = store[name]
        sims = idx["vectors"] @ q
        return sims, idx["meta"]

    if mode in ("ops_only", "ops"):
        sims, meta = _score_index("ops")
        idx = np.argsort(-sims)[:k]
        results = [meta[i] | {"score": float(sims[i]), "index": "ops"} for i in idx]

    elif mode in ("customer_only", "customer"):
        sims, meta = _score_index("customer")
        idx = np.argsort(-sims)[:k]
        results = [
            meta[i] | {"score": float(sims[i]), "index": "customer"} for i in idx
        ]

    else:  # blended
        tmp: Dict[str, Dict] = {}
        for name in ("ops", "customer"):
            sims, meta = _score_index(name)
            idx = np.argsort(-sims)[:k]
            for i in idx:
                m = meta[i]
                rid = m.get("id", f"{name}-{i}")
                cand = m | {"score": float(sims[i]), "index": name}
                if rid not in tmp or cand["score"] > tmp[rid]["score"]:
                    tmp[rid] = cand

        results = list(tmp.values())
        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:k]

    return results


# -------------------------------------------------------------------
# Session management
# -------------------------------------------------------------------
def ensure_ai_session():
    now = time.time()
    started_at = st.session_state.get("ai_session_started_at")

    if started_at is None or (now - started_at) > SESSION_TTL_SECONDS:
        st.session_state["ai_session_started_at"] = now
        st.session_state["chat_history"] = []


# -------------------------------------------------------------------
# View suggestion / routing
# -------------------------------------------------------------------
TAB_LABELS = {
    "overview": "📊 Overview",
    "store": "🏪 Store Explorer",
    "forecast": "🔬 Forecast Lab",
    "events": "📋 Event & Seasonality",
    "rosters_tm": "👤 Rosters - Individual TM",
    "rosters_week": "📅 Rosters - Weekly Coverage & Scheduling",
}


def suggest_view(query: str) -> Dict:
    q = query.lower()

    if any(w in q for w in ["roster", "shift", "role", "tm", "hours worked", "host"]):
        key = "rosters_tm"
        reason = "This looks like a question about individual team member hours or wages."
    elif any(w in q for w in ["coverage", "weekly", "week of year", "pattern", "12-week"]):
        key = "rosters_week"
        reason = "Weekly coverage charts will show the pattern across weeks."
    elif any(w in q for w in ["black friday", "christmas", "event", "seasonality", "campaign"]):
        key = "events"
        reason = "The Event & Seasonality tab lets you slice by event type and date."
    elif any(w in q for w in ["hybrid", "randomforest", "random forest", "intuitive", "forecast lab", "model"]):
        key = "forecast"
        reason = "Forecast Lab is where you experiment with models and accuracy."
    elif any(
        w in q
        for w in ["double bay", "castle", "parramatta", "sales per labour", "compare stores", "store kpi"]
    ):
        key = "store"
        reason = "Store Explorer shows per-store KPIs, charts and comparisons."
    else:
        key = "overview"
        reason = "The Overview tab summarises network-level performance."

    return {"tab_key": key, "tab_label": TAB_LABELS[key], "reason": reason}


# -------------------------------------------------------------------
# Prompt construction & chat
# -------------------------------------------------------------------
def build_system_prompt(response_mode: str, deep_dive: bool) -> str:
    base = (
        "You are Mecca AI, a senior data & forecasting analyst for a retail network. "
        "You understand store-level sales, forecast models (MECCA, Hybrid, RandomForest, Intuitive), "
        "labour hours, wages and rostering, as well as customer reviews, competitor reviews and news "
        "about Mecca. Use the numeric and textual context snippets carefully and do not invent "
        "specific numbers if they are not present.\n"
    )

    if response_mode == "short":
        length = (
            "Start with a heading 'Key takeaways' followed by 3–5 bullet points "
            "(each at most 20 words). After that, add at most one short paragraph "
            "to give a bit more explanation."
        )
    else:
        length = (
            "Start with a heading 'Key takeaways' and 3–6 bullet points (≤25 words each). "
            "Then add three short sections with Markdown headings: "
            "'What's happening', 'Why it matters', and 'Recommended actions'."
        )

    if deep_dive:
        depth = (
            "In 'Recommended actions', be specific about operational changes, and comment on drivers, "
            "outliers and model behaviour where relevant. When review or competitor snippets are present, "
            "tie actions explicitly to customer experience and what competitors do well."
        )
    else:
        depth = (
            "Keep the tone business-friendly and focus on the story and implications rather than "
            "technical modelling details."
        )

    return base + length + " " + depth


def format_context_snippets(snippets: List[Dict]) -> str:
    parts = []
    for s in snippets:
        label = s.get("type", s.get("index", "snippet"))
        parts.append(f"[{label}] {s['text']}")
    return "\n".join(parts)


def ask_ai(
    user_query: str,
    response_mode: str = "short",
    deep_dive: bool = False,
    max_ctx_snippets: int = 8,
    retrieval_mode: str = "blended",  # "blended" | "ops_only" | "customer_only"
) -> Tuple[str, List[Dict], Dict]:
    """
    Main entry point: run retrieval + chat and update Streamlit session history.

    Returns:
        answer_text, snippets_used, view_suggestion_dict
    """
    ensure_ai_session()

    view_info = suggest_view(user_query)

    retrieved = search_index(user_query, k=max_ctx_snippets, mode=retrieval_mode)
    ctx_text = format_context_snippets(retrieved)

    sys_prompt = build_system_prompt(response_mode, deep_dive)
    messages = [{"role": "system", "content": sys_prompt}]

    if ctx_text:
        messages.append(
            {
                "role": "system",
                "content": "Context from Mecca data (not exhaustive):\n" + ctx_text,
            }
        )

    # Instruct the model to help users navigate the app
    messages.append(
        {
            "role": "system",
            "content": (
                "Always answer in Markdown. At the end of your answer, add a short section "
                "titled 'Where to explore in the app' with 1–2 bullet points explaining "
                f"why the **{view_info['tab_label']}** tab is the best next place to look at charts "
                "for this question."
            ),
        }
    )

    # Short conversation memory
    history = st.session_state.get("chat_history", [])
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_query})

    max_tokens = 350 if response_mode == "short" else 900

    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3 if deep_dive else 0.4,
    )

    answer = resp.choices[0].message.content

    # Update history
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": answer})
    st.session_state["chat_history"] = history

    return answer, retrieved, view_info

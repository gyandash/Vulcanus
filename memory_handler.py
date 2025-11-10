# src/core/memory_handler.py
"""
Memory handler with semantic search + time-aware insights.

Behavior:
- Stores per-user memories in JSON at ../data/user_memory.json
- Provides CRUD: add_memory, get_memory, clear_memory
- Provides semantic search (embeddings or TF-IDF fallback)
- Provides time-aware summaries ("You last updated X 3 days ago.")
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

# -------------------------------
# File paths and data management
# -------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MEMORY_FILE = os.path.join(DATA_DIR, "user_memory.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_memory() -> Dict[str, Any]:
    """Load all memory data from JSON file."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_memory(memory: Dict[str, Any]):
    """Save memory dictionary to JSON file."""
    _ensure_data_dir()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


# -------------------------------
# CRUD Functions
# -------------------------------

def add_memory(username: str, key: str, value: str):
    """Add or update a memory entry with timestamp."""
    memory = _load_memory()
    if username not in memory:
        memory[username] = {}
    memory[username][key] = {"value": value, "timestamp": datetime.utcnow().isoformat()}
    _save_memory(memory)


def get_memory(username: str, key: str = None):
    """Retrieve all user memories or a single memory key."""
    memory = _load_memory()
    if username not in memory:
        return None
    if key:
        return memory[username].get(key)
    return memory[username]


def clear_memory(username: str):
    """Remove all stored memories for a specific user."""
    memory = _load_memory()
    if username in memory:
        del memory[username]
        _save_memory(memory)


# -------------------------------
# Semantic Search Utilities
# -------------------------------
# Tries sentence-transformers first, falls back to TF-IDF.
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from numpy.linalg import norm

    _HAS_S_TRANSFORMERS = True
    _DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    def _embed_texts_sbert(texts: List[str], model_name: str = _DEFAULT_MODEL_NAME) -> List[List[float]]:
        model = SentenceTransformer(model_name)
        return model.encode(texts, show_progress_bar=False).tolist()

    def _cosine_sim(a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        if norm(a) == 0 or norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (norm(a) * norm(b)))

except Exception:
    _HAS_S_TRANSFORMERS = False

# Fallback TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    import numpy as np
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def semantic_search(username: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Return top_k memory entries most semantically similar to query.
    Each result: {"key": str, "value": str, "timestamp": str, "score": float}
    """
    user_mem = get_memory(username)
    if not user_mem:
        return []

    keys, texts, timestamps = [], [], []
    for k, meta in user_mem.items():
        keys.append(k)
        v = meta.get("value", "")
        texts.append(v if isinstance(v, str) else str(v))
        timestamps.append(meta.get("timestamp"))

    # Try SBERT embeddings
    if _HAS_S_TRANSFORMERS:
        try:
            text_emb = _embed_texts_sbert(texts)
            query_emb = _embed_texts_sbert([query])[0]
            scores = [_cosine_sim(query_emb, emb) for emb in text_emb]
            ranked = sorted(
                [{"key": keys[i], "value": texts[i], "timestamp": timestamps[i], "score": scores[i]}
                 for i in range(len(keys))],
                key=lambda x: x["score"],
                reverse=True
            )
            return ranked[:top_k]
        except Exception:
            pass  # fallback to TF-IDF

    # Try TF-IDF
    if _HAS_SKLEARN:
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts + [query])
            query_vec = tfidf_matrix[-1]
            doc_matrix = tfidf_matrix[:-1]
            cosine_similarities = linear_kernel(query_vec, doc_matrix).flatten()
            ranked_idx = cosine_similarities.argsort()[::-1]
            results = []
            for idx in ranked_idx[:top_k]:
                results.append({
                    "key": keys[idx],
                    "value": texts[idx],
                    "timestamp": timestamps[idx],
                    "score": float(cosine_similarities[idx])
                })
            return results
        except Exception:
            pass

    # Last resort: substring match
    q_lower = query.lower()
    results = []
    for i, txt in enumerate(texts):
        score = 1.0 if q_lower in txt.lower() or q_lower in keys[i].lower() else 0.0
        results.append({"key": keys[i], "value": texts[i], "timestamp": timestamps[i], "score": score})
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# -------------------------------
# Time-Aware Learning Summaries
# -------------------------------

def get_time_aware_summary(username: str) -> List[str]:
    """
    Generates human-readable summaries of how recent each memory update was.
    Example:
      "You last updated 'project_name' (Student Database System) 3 days ago."
    """
    data = get_memory(username)
    if not data:
        return ["No memories found for this user yet."]

    summaries = []
    now = datetime.utcnow()

    for key, meta in data.items():
        ts = meta.get("timestamp")
        value = meta.get("value")
        if not ts:
            continue
        try:
            ts_dt = datetime.fromisoformat(ts)
            days = (now - ts_dt).days

            if days == 0:
                time_phrase = "today"
            elif days == 1:
                time_phrase = "yesterday"
            else:
                time_phrase = f"{days} days ago"

            summaries.append(f"🕒 You last updated '{key}' ({value}) {time_phrase}.")
        except Exception:
            summaries.append(f"🕒 You last updated '{key}' recently.")
    return summaries
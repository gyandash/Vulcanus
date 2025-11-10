"""
Memory handler with semantic search + visual scoring enhancements.

Behavior:
- Stores per-user memories in JSON at ../data/user_memory.json
- Provides basic CRUD: add_memory, get_memory, clear_memory
- Provides semantic search with color-coded confidence and auto-filtering.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MEMORY_FILE = os.path.join(DATA_DIR, "user_memory.json")

# -------------------------------------------------------------------
# --- Basic File Management Helpers ---
# -------------------------------------------------------------------
def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_memory() -> Dict[str, Any]:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_memory(memory: Dict[str, Any]):
    _ensure_data_dir()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


# -------------------------------------------------------------------
# --- CRUD Operations ---
# -------------------------------------------------------------------
def add_memory(username: str, key: str, value: str):
    memory = _load_memory()
    if username not in memory:
        memory[username] = {}
    memory[username][key] = {"value": value, "timestamp": datetime.utcnow().isoformat()}
    _save_memory(memory)


def get_memory(username: str, key: str = None):
    memory = _load_memory()
    if username not in memory:
        return None
    if key:
        return memory[username].get(key)
    return memory[username]


def clear_memory(username: str):
    memory = _load_memory()
    if username in memory:
        del memory[username]
        _save_memory(memory)


# -------------------------------------------------------------------
# --- Semantic Search Utilities ---
# -------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from numpy.linalg import norm

    _HAS_S_TRANSFORMERS = True
    _DEFAULT_MODEL_NAME = "all-mpnet-base-v2"  # 🔥 High-quality embedding model (SBERT)

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

# TF-IDF fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    import numpy as np
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# -------------------------------------------------------------------
# --- Semantic Search Function ---
# -------------------------------------------------------------------
def semantic_search(username: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Return top_k memory entries most similar to query.
    Each result: {"key", "value", "timestamp", "score", "color", "label"}
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

    # --- SBERT Path ---
    if _HAS_S_TRANSFORMERS:
        try:
            text_embeddings = _embed_texts_sbert(texts)
            query_embedding = _embed_texts_sbert([query])[0]
            scores = [_cosine_sim(query_embedding, emb) for emb in text_embeddings]

            # Build results
            results = [
                {
                    "key": keys[i],
                    "value": texts[i],
                    "timestamp": timestamps[i],
                    "score": round(float(scores[i]), 3),
                }
                for i in range(len(keys))
            ]

            # Filter + color-code
            filtered = [r for r in results if r["score"] >= 0.4]
            for r in filtered:
                if r["score"] >= 0.8:
                    r["color"] = "#00FF99"  # strong match (green)
                    r["label"] = "High match"
                elif r["score"] >= 0.6:
                    r["color"] = "#FFD700"  # moderate (yellow)
                    r["label"] = "Medium match"
                else:
                    r["color"] = "#FF7F50"  # low confidence (orange)
                    r["label"] = "Low match"

            filtered.sort(key=lambda x: x["score"], reverse=True)
            return filtered[:top_k]

        except Exception:
            pass

    # --- TF-IDF fallback ---
    if _HAS_SKLEARN:
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts + [query])
            query_vec = tfidf_matrix[-1]
            doc_matrix = tfidf_matrix[:-1]
            cosine_similarities = linear_kernel(query_vec, doc_matrix).flatten()
            results = []
            for i, score in enumerate(cosine_similarities):
                if score >= 0.4:  # filter weak matches
                    results.append({
                        "key": keys[i],
                        "value": texts[i],
                        "timestamp": timestamps[i],
                        "score": round(float(score), 3),
                        "color": "#FFD700" if score < 0.7 else "#00FF99",
                        "label": "Moderate match" if score < 0.7 else "Strong match"
                    })
            return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        except Exception:
            pass

    # --- Fallback substring match ---
    lower_q = query.lower()
    results = []
    for i, txt in enumerate(texts):
        score = 1.0 if lower_q in txt.lower() or lower_q in keys[i].lower() else 0.0
        if score >= 0.4:
            results.append({
                "key": keys[i],
                "value": texts[i],
                "timestamp": timestamps[i],
                "score": round(score, 3),
                "color": "#FFD700" if score < 0.7 else "#00FF99",
                "label": "Exact match" if score == 1.0 else "Weak match"
            })
    return results[:top_k]

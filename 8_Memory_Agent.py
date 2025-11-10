# src/pages/8_Memory_Agent.py
import streamlit as st
from datetime import datetime
from core.agents import MemoryAgent
from core.memory_handler import get_memory, get_time_aware_summary  # ⬅️ NEW import

# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 Context-Aware Memory Agent",
    layout="wide"
)

# --- Global Page Styling (same glass background as Multi-Agent Collab) ---
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle at top left, #1b1b2f, #121212);
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, rgba(15,15,25,0.95), rgba(25,25,40,0.97));
        backdrop-filter: blur(12px);
    }
    h1, h2, h3 {
        color: #CBB9FF !important;
        text-shadow: 0px 0px 10px rgba(120,90,255,0.4);
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #CCCCCC;
    }
    .memory-box {
        background: rgba(60, 60, 85, 0.25);
        padding: 15px 20px;
        border-radius: 12px;
        border: 1px solid rgba(150, 130, 255, 0.2);
        box-shadow: 0 0 15px rgba(100, 80, 255, 0.1);
        margin-top: 10px;
        line-height: 1.7;
        color: #E6E6E6;
        font-size: 15px;
    }
    .stTextInput > div > div > input, .stTextArea textarea {
        background-color: rgba(30,30,45,0.8);
        color: #E0E0E0;
        border-radius: 8px;
        border: 1px solid rgba(150,130,255,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header Section ---
st.markdown(
    """
    <h1 style='text-align:center; color:#CBB9FF;'>🧠 Context-Aware Memory Agent</h1>
    <p style='text-align:center; color:#B0B0B0; font-size:16px;'>
        Your AI assistant that remembers your preferences, past tasks, and project context 
        to make Vulcanus smarter and more personalized.
    </p>
    """,
    unsafe_allow_html=True
)

# --- Initialize Agent ---
username = st.session_state.get("username", "guest")
agent = MemoryAgent(username)

st.markdown("---")

# ============================================================
# 💾 SECTION 1 — ADD MEMORY
# ============================================================
st.markdown("### 💾 Add New Memory")
st.markdown("<p style='color:gray;'>Store your preferences or project details for later use.</p>", unsafe_allow_html=True)

key = st.text_input("🗝️ Enter a key (e.g., project_name, preferred_language):")
value = st.text_area("🧩 Enter the value (what to remember):")

if st.button("💾 Save Memory", use_container_width=True):
    if key and value:
        st.success(agent.remember(key, value))
    else:
        st.warning("⚠️ Please provide both key and value before saving.")

st.markdown("---")

# ============================================================
# 🔍 SECTION 2 — BASIC RECALL
# ============================================================
st.markdown("### 🔍 Recall Memory")
st.markdown("<p style='color:gray;'>Retrieve your saved context. Leave blank to list all.</p>", unsafe_allow_html=True)

key_query = st.text_input("Enter key to recall (optional):")
if st.button("🔎 Recall", use_container_width=True):
    result = agent.recall(key_query)
    if result.startswith("⚠️"):
        st.warning(result)
    elif result.startswith("🧠"):
        st.success(result)
    else:
        html = ""
        for line in result.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                html += f"• <b style='color:#CBB9FF;'>{k.strip()}</b> → <span style='color:#7DEFFF;'>{v.strip()}</span><br>"
            else:
                html += line + "<br>"
        st.markdown(f"<div class='memory-box'>{html}</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 🤖 SECTION 3 — SEMANTIC (AI) RECALL
# ============================================================
st.markdown("### 🤖 Semantic (AI) Recall")
st.markdown("<p style='color:gray;'>Search memories by meaning (try “my last project” or “preferred backend”).</p>", unsafe_allow_html=True)

semantic_query = st.text_input("Enter a natural-language query:")
if st.button("🧠 AI Recall", use_container_width=True):
    if not semantic_query.strip():
        st.warning("⚠️ Please enter a query before searching.")
    else:
        results = agent.semantic_recall(semantic_query, top_k=5)
        if not results:
            st.info("No related memories found.")
        else:
            html = ""
            for r in results:
                ts, score = r.get("timestamp", ""), r.get("score", 0)
                # Human-friendly timestamp
                ts_disp = ""
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        diff = (datetime.utcnow() - dt).days
                        ts_disp = "🕒 today" if diff == 0 else f"🕒 {diff} day{'s' if diff!=1 else ''} ago"
                    except Exception:
                        ts_disp = ts
                html += f"""
                <div style='padding:8px 0;border-bottom:1px solid rgba(120,120,200,0.1);'>
                    <b style='color:#CBB9FF;'>{r['key']}</b> 
                    <small style='color:#9fdfff;'>(score {score:.3f})</small><br>
                    <span style='color:#E6E6E6;'>{r['value']}</span><br>
                    <small style='color:#9fdfff;'>{ts_disp}</small>
                </div>"""
            st.markdown(f"<div class='memory-box'>{html}</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 🧹 SECTION 4 — CLEAR MEMORY
# ============================================================
st.markdown("### 🧹 Clear Memory")
st.markdown("<p style='color:gray;'>Delete all saved memories for this user.</p>", unsafe_allow_html=True)
if st.button("🧹 Clear All Memory", use_container_width=True):
    st.warning(agent.clear_all())

st.markdown("---")

# ============================================================
# 📊 SECTION 5 — SUMMARY & TIME INSIGHTS
# ============================================================
st.markdown("### 📊 Current Memory Summary & Time Insights")
st.markdown("<p style='color:gray;'>View all stored memories and their last update time.</p>", unsafe_allow_html=True)

summary = agent.recall()
if "⚠️ No memory found yet" not in summary:
    formatted = ""
    data = get_memory(username)
    for line in summary.split("\n"):
        if ":" not in line:
            formatted += line + "<br>"
            continue
        k, v = [x.strip() for x in line.split(":", 1)]
        ts = data.get(k, {}).get("timestamp")
        time_info = ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                days = (datetime.utcnow() - dt).days
                if days == 0:
                    time_info = "🕒 updated today"
                elif days == 1:
                    time_info = "🕒 updated 1 day ago"
                else:
                    time_info = f"🕒 updated {days} days ago"
            except Exception:
                time_info = ""
        formatted += f"• <b style='color:#CBB9FF;'>{k}</b> → <span style='color:#7DEFFF;'>{v}</span> <small style='color:#9fdfff;'>{time_info}</small><br>"
    st.markdown(f"<div class='memory-box'>{formatted}</div>", unsafe_allow_html=True)
else:
    st.info("No memories yet. Add one above! 💡")

# ============================================================
# 🕒 SECTION 6 — TIME-AWARE LEARNING SUMMARY
# ============================================================
st.markdown("---")
st.markdown("### 🕒 Time-Aware Learning Summary")
st.markdown("<p style='color:gray;'>See when each memory was last updated, e.g., “You last worked on X 3 days ago.”</p>", unsafe_allow_html=True)

time_summaries = get_time_aware_summary(username)
if time_summaries:
    html = "".join(f"<div style='margin-bottom:6px;color:#E6E6E6;'>{s}</div>" for s in time_summaries)
    st.markdown(f"<div class='memory-box'>{html}</div>", unsafe_allow_html=True)
else:
    st.info("No time-based summaries available yet. Save some memories first!")
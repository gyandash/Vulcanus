# src/pages/multi_agent_collab.py
import streamlit as st
from datetime import datetime
import uuid
import asyncio

# Core imports
from core.agents import (
    get_ragbits_llm_client,
    RagbitsCodeGenerationAgent,
    RagbitsDataLineageAgent,
    RagbitsWireframeAgent,
    RagbitsCloudCodeConverterAgent,
    MemoryAgent,
)
from core.multi_agent_orchestrator import ConversationOrchestrator
from core.neo4j_handler import Neo4jHandler
from core.memory_handler import get_memory

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="🤝 Multi-Agent Collaboration", layout="wide")

# ------------------ GLOBAL STYLES ------------------
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #121231, #050510);
        color: #EEE;
        font-family: 'Poppins', sans-serif;
    }
    .title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #cbb9ff;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.1rem;
        color: #aaa;
        text-align: center;
        margin-bottom: 30px;
    }
    .context-box {
        background: rgba(80,80,130,0.2);
        border: 1px solid rgba(160,160,255,0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 25px;
        box-shadow: 0 0 10px rgba(0,0,0,0.25);
        line-height: 1.8;
    }
    .agent-box {
        background-color: rgba(100,100,150,0.15);
        border: 1px solid rgba(120,120,180,0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    code {
        background-color: #1e1e2f !important;
        border-radius: 8px;
        padding: 8px !important;
        color: #d1c7ff !important;
        font-size: 0.9rem !important;
    }
    pre code {
        box-shadow: 0 0 10px rgba(130,100,255,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ PAGE HEADER ------------------
st.markdown("<h1 class='title'>🤝 Multi-Agent Collaboration Simulator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Let AI agents with different expertise collaborate to design systems and solve problems efficiently.</p>", unsafe_allow_html=True)

# ------------------ USER & MEMORY CONTEXT ------------------
username = st.session_state.get("username", "guest")
if not isinstance(username, str):
    username = str(username)

memory_agent = MemoryAgent(username)
mem_data = get_memory(username)
suggestion = memory_agent.suggest()

# --- Helper: Render memory context ---
def render_memory_context():
    """Show loaded user memory in bullet-style format."""
    if mem_data:
        formatted_context = ""
        for k, v in mem_data.items():
            val = v.get("value", "")
            formatted_context += f"• <b style='color:#CBB9FF;'>{k}</b>: <span style='color:#7DEFFF;'>{val}</span><br>"
        st.markdown(f"""
            <div class='context-box'>
                <h3 style="color:#CBB9FF;">📚 Loaded Memory Context</h3>
                {formatted_context}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No stored memory found. Add preferences in the Memory Agent first!")

# --- Helper: Render smart suggestions ---
def render_suggestions():
    """Show AI-powered suggestions."""
    if suggestion and "No stored memory" not in suggestion:
        suggestions = [s.strip() for s in suggestion.replace("🧠 Suggested context:", "").split("|")]
        formatted_suggestions = ""
        for s in suggestions:
            if s:
                formatted_suggestions += f"• <span style='color:#9FDFFF;'>{s}</span><br>"
        st.markdown(f"""
            <div class='context-box' style="background:rgba(60,60,90,0.3);">
                <h3 style="color:#9FDFFF;">🧠 Smart Suggestion</h3>
                {formatted_suggestions}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("💡 No personalized suggestions yet. Add preferences in your Memory Agent.")

# --- Helper: Render user context (fixed issue) ---
def render_user_context(context: dict):
    """Render current active user context cleanly in bullet points."""
    if not context:
        return
    formatted_context = ""
    for key, value in context.items():
        formatted_context += f"• <b style='color:#CBB9FF;'>{key}</b>: <span style='color:#7DEFFF;'>{value}</span><br>"
    st.markdown(f"""
        <div class='context-box' style="background:rgba(50,50,90,0.3);">
            <h3 style="color:#B0E0FF;">👤 User Context</h3>
            {formatted_context}
        </div>
    """, unsafe_allow_html=True)

# --- Render memory and suggestion at top ---
render_memory_context()
render_suggestions()

# ------------------ UI INPUTS ------------------
st.markdown("### 1️⃣ Define the high-level goal or problem")
goal = st.text_area(
    "Describe what you want the agents to work on",
    height=120,
    placeholder="e.g. Build a To-Do app using React frontend, Flask backend, and PostgreSQL."
)

st.markdown("### 2️⃣ Configure Agents and Collaboration Settings")

agent_options = ["CodeAgent", "DataAgent", "WireframeAgent", "InfraAgent"]
selected_agents = st.multiselect("Select Agents to Include", agent_options, default=["CodeAgent", "DataAgent"])

rounds = st.slider("Rounds of Collaboration", 1, 5, 2)
temperature = st.slider("AI Creativity (Temperature)", 0.1, 1.0, 0.6)
persona = st.selectbox("Persona for CodeAgent",
                       ["Standard", "Senior Engineer (Robust)", "Creative Engineer", "Minimalist Developer"])

col_run, col_clear, col_save = st.columns([1, 1, 1])

# ------------------ SESSION STATE ------------------
if "transcript" not in st.session_state:
    st.session_state.transcript = []

def clear_transcript():
    st.session_state.transcript = []
    st.success("Transcript cleared!")

# ------------------ AUTO-SUMMARIZER ------------------
async def summarize_transcript(transcript_text: str) -> str:
    """Uses LLM to generate a concise summary of collaboration."""
    try:
        llm_client = get_ragbits_llm_client()
        prompt = f"""
        Summarize this collaboration in one short sentence focusing on the project goal and tech stack:

        {transcript_text}
        """
        response = await llm_client.generate(prompt=prompt)
        return response.strip()
    except Exception as e:
        return f"Summary unavailable: {e}"

# ------------------ RUN COLLABORATION ------------------
if col_run.button("▶️ Run Collaboration", use_container_width=True):
    if not goal.strip():
        st.warning("Please enter a goal first.")
    else:
        with st.spinner("🤖 Agents are collaborating... please wait."):
            llm_client = get_ragbits_llm_client()
            agents = []
            for name in selected_agents:
                if name == "CodeAgent":
                    agents.append(RagbitsCodeGenerationAgent(llm_client, persona))
                elif name == "DataAgent":
                    agents.append(RagbitsDataLineageAgent(llm_client))
                elif name == "WireframeAgent":
                    agents.append(RagbitsWireframeAgent(llm_client))
                elif name == "InfraAgent":
                    agents.append(RagbitsCloudCodeConverterAgent(llm_client))

            # ✅ Combine username + memory context for rendering
            user_context = {"username": username}
            if mem_data:
                for k, v in mem_data.items():
                    user_context[k] = v.get("value", "")

            render_user_context(user_context)

            orchestrator = ConversationOrchestrator(agents, goal, context=user_context)
            orchestrator.run(rounds)
            st.session_state.transcript = orchestrator.get_log()

            # ✅ Auto-save context to memory
            memory_agent.remember("last_project", goal)
            selected_stack = ", ".join(selected_agents)
            memory_agent.remember("preferred_stack", selected_stack)
            memory_agent.remember("last_updated", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

            # ✅ Auto-summarize
            transcript_text = "\n".join([f"{m['agent']}: {m['message']}" for m in st.session_state.transcript])
            summary = asyncio.run(summarize_transcript(transcript_text))
            memory_agent.remember("last_summary", summary)

            st.rerun()

if col_clear.button("🧹 Clear Transcript", use_container_width=True):
    clear_transcript()

# ------------------ SAVE TO NEO4J ------------------
if col_save.button("💾 Save to Neo4j", use_container_width=True):
    if not st.session_state.transcript:
        st.warning("No transcript found to save.")
    else:
        try:
            handler = Neo4jHandler()
            event_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            handler.store_project_flow_event(
                event_id, goal, str(st.session_state.transcript), {}, timestamp, diagram_type="MultiAgent"
            )
            handler.close()
            st.success("✅ Collaboration saved to Neo4j successfully!")
        except Exception as e:
            st.error(f"Error saving to Neo4j: {e}")

# ------------------ DISPLAY TRANSCRIPT ------------------
if st.session_state.transcript:
    st.markdown("---")
    st.markdown("### 3️⃣ Agent Conversation Transcript")

    grouped = {}
    for msg in st.session_state.transcript:
        grouped.setdefault(msg["round"], []).append(msg)

    for round_num, messages in grouped.items():
        with st.expander(f"🌀 Round {round_num}", expanded=True):
            for msg in messages:
                agent_name = msg["agent"]
                message_content = msg["message"]

                st.markdown(
                    f"""
                    <div class='agent-box'>
                        <h4 style="margin-bottom:8px; color:#d0b3ff;">🧠 {agent_name}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if "```" in message_content:
                    code_blocks = message_content.split("```")
                    for i in range(1, len(code_blocks), 2):
                        code_part = code_blocks[i].strip()
                        if "\n" in code_part:
                            lang = code_part.split("\n", 1)[0].strip()
                            code_body = code_part[len(lang):].strip()
                        else:
                            lang, code_body = "text", code_part
                        st.code(code_body, language=lang or "text")
                else:
                    st.markdown(message_content)

    # ------------------ DOWNLOAD TXT ONLY ------------------
    st.markdown("---")
    st.markdown("### 💾 Download Transcript")
    transcript_text = "\n".join([
        f"Round {m['round']} | {m['agent']}:\n{m['message']}\n{'-'*60}"
        for m in st.session_state.transcript
    ])
    st.download_button(
        "📥 Download Transcript (.txt)",
        transcript_text,
        file_name="collaboration_log.txt",
        mime="text/plain",
        use_container_width=True,
    )
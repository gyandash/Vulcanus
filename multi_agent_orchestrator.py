# src/core/multi_agent_orchestrator.py
from typing import List, Dict, Any
import streamlit as st
from core.agents import MemoryAgent  # ✅ Added for context-aware integration


class ConversationOrchestrator:
    """
    Multi-agent orchestrator with context-awareness.
    Each agent should implement:
      - propose_action(goal: str, context: dict) -> str
      - receive_message(sender_name: str, message: str, context: dict) -> None

    This version integrates Feature #6 (Memory Agent) to inject
    user preferences, past tasks, and project context automatically.
    """

    def __init__(self, agents: List[Any], goal: str, username: str = "guest", context: dict = None):
        self.agents = agents
        self.goal = goal
        self.username = username
        self.context = context or {}
        self.transcript = []  # List of {"round": int, "agent": name, "message": str}

        # ✅ Initialize MemoryAgent to load user memory
        self.memory_agent = MemoryAgent(username)
        self._load_user_context()

    def _load_user_context(self):
        """
        Loads stored user memory (Feature #6) and injects it
        into the orchestrator context for all agents to use.
        """
        stored_memory = self.memory_agent.recall()
        if "⚠️" not in stored_memory:
            st.info("📚 Loaded context from memory:")
            st.markdown(
                f"<pre style='color:#CBB9FF; font-size:14px; background:rgba(40,40,60,0.25); padding:10px; border-radius:8px;'>{stored_memory}</pre>",
                unsafe_allow_html=True,
            )

            for line in stored_memory.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    self.context[k.strip()] = v.strip()

            # Show smart suggestion
            suggestion = self.memory_agent.suggest()
            st.markdown(
                f"""
                <div style="
                    background:rgba(60,60,90,0.25);
                    border:1px solid rgba(150,130,255,0.2);
                    padding:12px;
                    border-radius:10px;
                    margin-top:8px;
                ">
                    <p style="color:#B0E0FF; font-size:14px;">{suggestion}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ No stored memory found. You can add preferences in the Memory Agent feature.")

    def run(self, rounds: int = 1):
        """
        Executes the multi-agent conversation flow.
        Automatically includes user memory context in the shared goal.
        """
        if not self.agents:
            st.error("No agents available to run.")
            return

        # 🧠 Inject user context directly into the goal
        if self.context:
            context_text = "\n\n### USER CONTEXT ###\n"
            for k, v in self.context.items():
                context_text += f"- {k}: {v}\n"
            self.goal += context_text

        # Run collaboration rounds
        for r in range(1, rounds + 1):
            st.markdown(f"### 🔁 Round {r}")
            for agent in self.agents:
                agent_name = getattr(agent, "persona", getattr(agent, "name", agent.__class__.__name__))
                try:
                    # Agent proposes an action
                    if hasattr(agent, "propose_action"):
                        msg = agent.propose_action(self.goal, self.context)
                    elif hasattr(agent, "propose"):
                        msg = agent.propose(self.goal, self.context)
                    elif hasattr(agent, "generate"):
                        msg = agent.generate(self.goal)
                    else:
                        msg = f"[{agent_name}] has no valid action method."
                except Exception as e:
                    msg = f"[{agent_name}] ERROR during propose: {e}"

                msg_text = str(msg or "").strip()
                self.transcript.append({"round": r, "agent": agent_name, "message": msg_text})

                # Display each round’s output
                with st.expander(f"🧩 {agent_name} says:", expanded=False):
                    st.markdown(f"<div style='color:#E0E0E0; font-size:15px;'>{msg_text}</div>", unsafe_allow_html=True)

                # Share message with other agents
                for other in self.agents:
                    if other is agent:
                        continue
                    try:
                        if hasattr(other, "receive_message"):
                            other.receive_message(agent_name, msg_text, self.context)
                        elif hasattr(other, "on_message"):
                            other.on_message(agent_name, msg_text, self.context)
                    except Exception:
                        # Ignore agent-side errors to keep orchestration stable
                        pass

    def get_log(self) -> List[Dict]:
        """Return conversation transcript."""
        return self.transcript

    def clear_log(self):
        """Clear stored transcript."""
        self.transcript = []
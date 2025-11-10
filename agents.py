# src/core/agents.py
import asyncio
import json
import os
import tempfile
from typing import AsyncGenerator, Iterable, Type, List, Dict
import shutil
import streamlit as st
from pydantic import BaseModel
from collections.abc import Iterable

# Ragbits Imports
from ragbits.agents import Agent
from ragbits.core.llms import LiteLLM
from ragbits.core.prompt import Prompt
from ragbits.agents import ToolCallResult

# Import get_ragbits_llm_client
from core.llm import get_ragbits_llm_client
from core.llm import WireframePromptInput, WireframePrompt

# -------------------------------------------------------------------
# --- Prompt Input Models (for base agents and specific tasks) ---
# -------------------------------------------------------------------

class AgentBaseInput(BaseModel):
    query: str = ""


class CodeGenerationPromptInput(BaseModel):
    original_code: str
    conversion_type: str
    user_instructions: str
    source_language: str = ""
    source_framework: str = ""
    target_language: str = ""
    target_framework: str = ""


class CodeGenerationPrompt(Prompt[CodeGenerationPromptInput, str]):
    system_prompt = """
    You are an AI assistant specialized in code generation, refactoring, optimization, and conversion.
    Your responses should be the code directly, without any conversational filler or explanation,
    unless explicitly asked for by the user instructions.
    Ensure the generated code is syntactically correct for the target language/framework.
    """
    user_prompt = """
    Operation: {{ conversion_type }}
    Original Code:
    ```
    {{ original_code }}
    ```
    {% if source_language and target_language %}
    Source Language: {{ source_language }} ({{ source_framework }})
    Target Language: {{ target_language }} ({{ target_framework }})
    {% endif %}
    User Instructions: {{ user_instructions }}
    Please provide the generated or converted code:
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.7
    llm_settings: LLMSettings = LLMSettings()


class DocumentQueryPromptInput(BaseModel):
    query: str
    context_str: str


class DocumentQueryPrompt(Prompt[DocumentQueryPromptInput, str]):
    system_prompt = """
    You are a highly accurate document question-answering assistant.
    Answer ONLY using the provided context.
    """
    user_prompt = """
    QUESTION:
    {{ query }}
    CONTEXT:
    {{ context_str }}
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.5
    llm_settings: LLMSettings = LLMSettings()


class DataLineagePromptInput(BaseModel):
    code_or_description: str


class DataLineagePrompt(Prompt[DataLineagePromptInput, str]):
    system_prompt = (
        "You are an AI assistant specialized in analyzing code or natural language descriptions "
        "to identify data sources, transformations, and sinks. Output valid JSON with nodes and edges."
    )
    user_prompt = """
    Analyze the following code or description and extract data lineage in JSON format:
    ```
    {{ code_or_description }}
    ```
    Provide ONLY the JSON output, wrapped in ```json...``` tags.
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.5
    llm_settings: LLMSettings = LLMSettings()


class CloudCodeConverterPromptInput(BaseModel):
    original_code: str
    file_type: str
    source_platform: str
    source_version: str
    target_platform: str
    target_version: str
    user_instructions: str = ""


class CloudCodeConverterPrompt(Prompt[CloudCodeConverterPromptInput, str]):
    system_prompt = (
        "You are an expert cloud code converter. Convert accurately between platforms and versions."
    )
    user_prompt = """
    Original Code:
    ```
    {{ original_code }}
    ```
    Source: {{ source_platform }} ({{ source_version }})
    Target: {{ target_platform }} ({{ target_version }})
    Instructions: {{ user_instructions }}
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.7
    llm_settings: LLMSettings = LLMSettings()


# -------------------------------------------------------------------
# --- Agent Definitions (Core + Multi-Agent)
# -------------------------------------------------------------------

class RagbitsCodeGenerationAgent(Agent):
    def __init__(self, llm: LiteLLM, persona: str = "Standard"):
        self.persona = persona
        class BaseAgentPrompt(Prompt[AgentBaseInput]):
            system_prompt = "You are a versatile AI assistant."
            user_prompt = "{{ query }}"
        super().__init__(llm=llm, prompt=BaseAgentPrompt)

    def generate_code(self, original_code: str, conversion_type: str,
                      user_instructions: str = "", source_language: str = "",
                      source_framework: str = "", target_language: str = "",
                      target_framework: str = "", temperature: float = 0.7) -> str:
        code_prompt_input_data = CodeGenerationPromptInput(
            original_code=original_code,
            conversion_type=conversion_type,
            user_instructions=user_instructions,
            source_language=source_language,
            source_framework=source_framework,
            target_language=target_language,
            target_framework=target_framework
        )
        code_gen_prompt_instance = CodeGenerationPrompt(code_prompt_input_data)
        code_gen_prompt_instance.llm_settings.temperature = temperature
        try:
            llm_client_for_this_call = get_ragbits_llm_client()
            response = asyncio.run(llm_client_for_this_call.generate(prompt=code_gen_prompt_instance))
            return response
        except Exception as e:
            return f"Error: {e}"

    def propose_action(self, goal, context):
        try:
            return self.generate_code(goal, "Generate", user_instructions=goal)
        except Exception as e:
            return f"[CodeAgent Error] {e}"

    def receive_message(self, sender_name, message, context):
        self.last_message = f"{sender_name}: {message}"


class RagbitsDataLineageAgent(Agent):
    def __init__(self, llm: LiteLLM):
        super().__init__(llm=llm, prompt=DataLineagePrompt)

    def extract_lineage(self, code_or_description: str) -> dict:
        lineage_prompt_input_data = DataLineagePromptInput(code_or_description=code_or_description)
        lineage_prompt_instance = DataLineagePrompt(lineage_prompt_input_data)
        try:
            llm_client_for_this_call = get_ragbits_llm_client()
            response_text = asyncio.run(llm_client_for_this_call.generate(prompt=lineage_prompt_instance))
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            data = json.loads(json_str)
            return data
        except Exception as e:
            st.error(f"Error in lineage extraction: {e}")
            return {"nodes": [], "edges": []}

    def propose_action(self, goal, context):
        try:
            lineage = self.extract_lineage(goal)
            return f"DataAgent: Identified {len(lineage.get('nodes', []))} nodes and {len(lineage.get('edges', []))} edges."
        except Exception as e:
            return f"[DataAgent Error] {e}"

    def receive_message(self, sender_name, message, context):
        self.last_message = f"{sender_name} -> {message}"


class RagbitsCloudCodeConverterAgent(Agent):
    def __init__(self, llm: LiteLLM):
        super().__init__(llm=llm, prompt=CloudCodeConverterPrompt)

    def convert_code(self, original_code: str, file_type: str, source_platform: str,
                     source_version: str, target_platform: str, target_version: str,
                     user_instructions: str = "", temperature: float = 0.7) -> str:
        conversion_prompt_input_data = CloudCodeConverterPromptInput(
            original_code=original_code,
            file_type=file_type,
            source_platform=source_platform,
            source_version=source_version,
            target_platform=target_platform,
            target_version=target_version,
            user_instructions=user_instructions
        )
        conversion_prompt_instance = CloudCodeConverterPrompt(conversion_prompt_input_data)
        conversion_prompt_instance.llm_settings.temperature = temperature
        try:
            llm_client_for_this_call = get_ragbits_llm_client()
            response = asyncio.run(llm_client_for_this_call.generate(prompt=conversion_prompt_instance))
            return response
        except Exception as e:
            return f"Error: {e}"

    def propose_action(self, goal, context):
        try:
            return self.convert_code(goal, "py", "AWS Lambda", "v1", "GCP", "v2")
        except Exception as e:
            return f"[InfraAgent Error] {e}"

    def receive_message(self, sender_name, message, context):
        self.last_message = f"{sender_name}: {message}"


class RagbitsWireframeAgent(Agent):
    def __init__(self, llm: LiteLLM):
        super().__init__(llm=llm, prompt=WireframePrompt)

    def generate_wireframe_code(self, user_description: str, mukuro_reference: str, temperature: float = 0.8) -> str:
        wireframe_prompt_input = WireframePromptInput(
            user_description=user_description,
            mukuro_reference=mukuro_reference
        )
        wireframe_prompt_instance = WireframePrompt(wireframe_prompt_input)
        wireframe_prompt_instance.llm_settings.temperature = temperature
        try:
            llm_client_for_this_call = get_ragbits_llm_client()
            response = asyncio.run(llm_client_for_this_call.generate(prompt=wireframe_prompt_instance))
            return response
        except Exception as e:
            return f"Error: {e}"

    def propose_action(self, goal, context):
        try:
            return f"WireframeAgent generated layout for: {goal[:100]}"
        except Exception as e:
            return f"[WireframeAgent Error] {e}"

    def receive_message(self, sender_name, message, context):
        self.last_message = f"{sender_name}: {message}"


# -------------------------------------------------------------------
# --- NEW: Context-Aware Memory Agent (Feature #6 with Semantic Recall)
# -------------------------------------------------------------------
from core.memory_handler import add_memory, get_memory, clear_memory, semantic_search

class MemoryAgent:
    """An agent that stores, recalls, suggests, and semantically searches user memories."""
    def __init__(self, username: str):
        self.username = username

    def remember(self, key: str, value: str):
        add_memory(self.username, key, value)
        return f"✅ Remembered: {key} → {value}"

    def recall(self, key: str = None):
        data = get_memory(self.username, key)
        if not data:
            return "⚠️ No memory found yet."
        if isinstance(data, dict):
            return "\n".join([f"{k}: {v['value']}" for k, v in data.items()])
        return f"🧠 Recalled: {data}"

    def clear_all(self):
        clear_memory(self.username)
        return "🧹 Memory cleared successfully!"

    def suggest(self, current_context: str = "") -> str:
        """Offers smart suggestions or autocompletion based on user preferences."""
        data = get_memory(self.username)
        if not data:
            return "💡 No stored memory yet. Start by adding your preferences!"

        suggestions = []
        if "preferred_language" in data:
            suggestions.append(f"Use {data['preferred_language']['value']} as your main language.")
        if "preferred_framework" in data:
            suggestions.append(f"Consider working with the {data['preferred_framework']['value']} framework.")
        if "last_project" in data:
            suggestions.append(f"Continue developing '{data['last_project']['value']}'.")
        if "project_name" in data:
            suggestions.append(f"Suggested project name: {data['project_name']['value']}.")

        if not suggestions:
            return "💡 Add more memories to improve suggestions!"

        return "🧠 Suggested context: " + " | ".join(suggestions)

    def semantic_recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """Performs AI-powered semantic recall using embeddings or TF-IDF."""
        if not query or not query.strip():
            return []
        results = semantic_search(self.username, query, top_k=top_k)
        return results
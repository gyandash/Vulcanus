# src/core/llm.py
import os
from dotenv import load_dotenv
import asyncio
from ragbits.core.llms import LiteLLM
from ragbits.core.prompt import Prompt
from pydantic import BaseModel
import json # Added for handling JSON output from AI

# Removed the import from core.agents here, as WireframePromptInput/WireframePrompt will be defined below.
# from core.agents import WireframePromptInput, WireframePrompt # THIS LINE IS REMOVED

load_dotenv()

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini/gemini-2.5-flash-preview-09-2025")

_ragbits_llm_client: LiteLLM = None

def get_ragbits_llm_client() -> LiteLLM:
    global _ragbits_llm_client
    if _ragbits_llm_client is None:
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in .env file. Please set it to proceed for LiteLLM.")
        try:
            _ragbits_llm_client = LiteLLM(model_name=GEMINI_MODEL_NAME)
            print(f"Ragbits LiteLLM client initialized successfully for model: {GEMINI_MODEL_NAME}!")
        except Exception as e:
            raise RuntimeError(f"Error initializing Ragbits LiteLLM client: {e}") from e
    return _ragbits_llm_client

# Define a proper BaseModel subclass for ChartPromptInput
class ChartPromptInput(BaseModel):
    data_preview: str
    user_query: str

# Define ChartPrompt subclass for proper prompt templating (UPDATED FOR ECHARTS)
class ChartPrompt(Prompt[ChartPromptInput, str]):
    system_prompt = """
    You are an expert data visualization assistant. Your task is to generate Python code
    that defines an ECharts `options` dictionary based on the provided data preview and user query.
    
    The output MUST define a single Python dictionary variable named `options_dict`,
    which represents the ECharts configuration, ready for direct use with `st_echarts(options=options_dict)`.
    
    Do NOT include any Streamlit `st.` calls, imports, or conversational filler.
    The `df` pandas DataFrame (your input data) is available in the execution scope. Use `df['Column'].tolist()`, `df.groupby()`, etc.
    
    Provide ONLY the Python code to define this `options_dict` variable, enclosed in a single '```python' block.
    Ensure the generated code is syntactically correct and directly runnable.
    
    **ECharts Data & Configuration Guidelines:**
    - **Basic Structure:** ECharts options typically have `title`, `legend`, `tooltip`, `xAxis`, `yAxis`, `series`.
    - **Data Access:** Convert pandas Series to lists for ECharts data, e.g., `df['Column'].tolist()`.
    - **Titles:** Use `title: {'text': 'Chart Title', 'left': 'center', 'textStyle': {'color': '#FFF'}}` for chart titles.
    - **Legends:** Use `legend: {'bottom': 'bottom', 'data': ['Series1', 'Series2'], 'textStyle': {'color': '#CCC'}}` for legends.
    - **Tooltips:** Always include `tooltip: {'trigger': 'axis'}` (for line/bar) or `'item'` (for pie/doughnut).
    - **Axis Labels/Lines:** For dark themes, set `axisLabel: {'color': '#CCC'}` and `axisLine: {'lineStyle': {'color': '#444'}}`.
    - **Grid Lines:** Set `splitLine: {'lineStyle': {'color': '#333'}}` in `yAxis` for dark grid lines.
    - **Bar/Line Charts:**
        - `xAxis`: `{'type': 'category', 'data': [...]}` or `{'type': 'value'}`.
        - `yAxis`: `{'type': 'value', 'name': 'Value Label'}` or `{'type': 'category', 'data': [...]}`.
        - `series`: `[{'name': 'Label', 'type': 'bar', 'data': [...], 'itemStyle': {'color': 'rgba(R,G,B,A)'}}]`.
        - For multiple bars on the same axis (side-by-side or stacked), add multiple series.
        - For stacked bars, add `'stack': 'total'` to each series in the stack.
    - **Pie/Doughnut Charts:**
        - `series`: `[{'name': 'Label', 'type': 'pie', 'radius': ['40%', '60%'], 'center': ['50%', '50%'], 'data': [{'value': X, 'name': 'Category'}...], 'label': {'show': False}, 'emphasis': {'itemStyle': {'shadowBlur': 10, 'shadowOffsetX': 0, 'shadowColor': 'rgba(0, 0, 0, 0.5)'}}}]`.
        - `radius` controls inner and outer size for doughnut (e.g., `['40%', '60%']`). For pie, use `['0%', '60%']`.
        - `data` for pie/doughnut charts is a list of dictionaries, e.g., `[{'value': 100, 'name': 'Category A'}, {'value': 200, 'name': 'Category B'}]`.
        - `label: {'show': True, 'formatter': '{b}: {d}%'}` for showing labels with percentage.
        - `itemStyle: {'borderColor': '#0e0d12', 'borderWidth': 2}` can make segments blend better with dark glassmorphic background.
    
    Example Output Format for a Bar Chart:
    ```python
    df_grouped = df.groupby('Product')['Sales'].sum().reset_index()
    options_dict = {
        "title": {"text": "Sales by Product", "left": "center", "textStyle": {"color": "#FFF"}},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": "bottom", "data": ["Total Sales"], "textStyle": {"color": "#CCC"}},
        "xAxis": {
            "type": "category",
            "data": df_grouped['Product'].tolist(),
            "axisLabel": {"color": "#CCC"},
            "axisLine": {"lineStyle": {"color": "#444"}}
        },
        "yAxis": {
            "type": "value",
            "name": "Total Sales",
            "nameTextStyle": {"color": "#CCC"},
            "axisLabel": {"color": "#CCC"},
            "splitLine": {"lineStyle": {"color": "#333"}}
        },
        "series": [
            {
                "name": "Total Sales",
                "type": "bar",
                "data": df_grouped['Sales'].tolist(),
                "itemStyle": {
                    "color": "rgba(255, 99, 132, 0.7)"
                }
            }
        ]
    }
    ```
    Example Output for a Doughnut Chart:
    ```python
    # Ensure 'Value' is a numeric column in your df or adapt this logic
    total_value = df['Sales'].sum() if 'Sales' in df.columns else 100 
    confidence_value = 0.75 * total_value # Example logic, replace with actual metric
    remaining_value = total_value - confidence_value
    options_dict = {
        "title": {"text": "AI Confidence Score", "left": "center", "textStyle": {"color": "#FFF"}},
        "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b}: {c} ({d}%)"},
        "legend": {"bottom": "bottom", "data": ["Confidence", "Remaining"], "textStyle": {"color": "#CCC"}},
        "series": [
            {
                "name": "Confidence Breakdown",
                "type": "pie",
                "radius": ["40%", "60%"], # Doughnut effect
                "center": ["50%", "50%"],
                "avoidLabelOverlap": False,
                "label": {
                    "show": False,
                    "position": "center"
                },
                "emphasis": {
                    "label": {
                        "show": True,
                        "fontSize": '20',
                        "fontWeight": 'bold',
                        "color": "#FFF"
                    }
                },
                "labelLine": {
                    "show": False
                },
                "data": [
                    {"value": confidence_value, "name": "Confidence", "itemStyle": {"color": "rgba(75, 192, 192, 0.7)"}},
                    {"value": remaining_value, "name": "Remaining", "itemStyle": {"color": "rgba(200, 200, 200, 0.3)"}}
                ],
                "itemStyle": {
                    "borderColor": "#0e0d12", # Background color to blend with theme
                    "borderWidth": 2
                }
            }
        ]
    }
    ```
    """
    user_prompt = """
    Given the following data preview from a pandas DataFrame `df`:
    ```
    {{ data_preview }}
    ```
    Generate Python code that defines an ECharts `options_dict` based on the user's request:
    "{{ user_query }}"
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.5
    llm_settings: LLMSettings = LLMSettings()

class ERDiagramPromptInput(BaseModel):
    description: str

class ERDiagramPrompt(Prompt[ERDiagramPromptInput, str]):
    system_prompt = """
    You are an expert in generating Mermaid.js Entity-Relationship (ER) diagram syntax based on descriptions.
    Provide ONLY the Mermaid ERD code. Do NOT include any conversational text or markdown code block wrappers.
    The generated code MUST start with 'erDiagram'.
    
    Ensure primary keys (PK), foreign keys (FK), and relationships (cardinality: ||--||, ||--o|, |o--o|, |o--||, |o--|o, etc., and identifying/non-identifying) are correctly inferred from the description.
    
    Relationship labels:
    - For single-word labels, use `entity ||--|{ entity : label`.
    - For multi-word labels, use `entity ||--|{ entity : "Multi-word label"`.
    - Avoid extra colons or unusual characters in link labels that could cause parsing errors in Mermaid.
    - Ensure valid cardinality symbols (e.g., ||, |o, }|, }o).
    
    Example Mermaid ERD:
    erDiagram
        CUSTOMER ||--o{ ORDER : has
        ORDER ||--|{ LINE_ITEM : contains
        PRODUCT }o--o{ LINE_ITEM : part_of
    """
    user_prompt = """
    Generate a Mermaid ER Diagram for the following description:
    "{{ description }}"
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.5
    llm_settings: LLMSettings = LLMSettings()

# NEW: Prompt for ER Diagram for multiple DataFrames
class ERDiagramMultiDFPromptInput(BaseModel):
    description: str
    df_schemas_json: str # JSON string of all DataFrame schemas: {'df1': ['col1', 'col2'], 'df2': ['colA', 'colB']}

class ERDiagramMultiDFPrompt(Prompt[ERDiagramMultiDFPromptInput, str]):
    system_prompt = """
    You are an expert in generating Mermaid.js Entity-Relationship (ER) diagram syntax that illustrates relationships between multiple data entities (DataFrames/tables).
    Your output MUST be ONLY the Mermaid ERD code. Do NOT include any conversational text, explanations, or markdown code block wrappers (e.g., ```mermaid).
    The generated code MUST start with 'erDiagram'.
    
    Analyze the provided descriptions and DataFrame schemas (including column names) to infer entities, their attributes, primary keys, and especially foreign key relationships between them.
    
    Guidelines for generating relationships:
    - For each DataFrame, represent it as a Mermaid entity, listing its columns as attributes. Clearly mark primary keys (PK) and foreign keys (FK).
    - If a column in one DataFrame strongly suggests a foreign key to another DataFrame's primary key (e.g., 'product_id' in 'orders' linking to 'id' or 'product_id' in 'products'), explicitly define that relationship.
    - Use clear cardinality symbols (e.g., `||--|{` for one-to-many, `||--||` for one-to-one, `}o--o{` for many-to-many).
    - Labels on relationships should be concise and descriptive (e.g., `ORDER ||--|{ CUSTOMER : "placed by"`).
    
    Example input schemas:
    {"products": ["product_id", "name", "price"], "orders": ["order_id", "customer_id", "product_id", "quantity"], "customers": ["customer_id", "first_name", "last_name"]}
    
    Example Output (based on above input):
    erDiagram
        products {
            string product_id PK
            string name
            float price
        }
        orders {
            string order_id PK
            string customer_id FK
            string product_id FK
            int quantity
        }
        customers {
            string customer_id PK
            string string last_name
        }
        products ||--o{ orders : "contains"
        customers ||--o{ orders : "places"
    
    """
    user_prompt = """
    Generate a Mermaid ER Diagram for the following conceptual description and DataFrame schemas:
    Description: "{{ description }}"
    DataFrame Schemas: {{ df_schemas_json }}
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.5
    llm_settings: LLMSettings = LLMSettings()

# NEW: Prompt for general Flow Diagram generation (Mermaid, PlantUML, Graphviz DOT)
class FlowDiagramPromptInput(BaseModel):
    description: str
    diagram_type: str # e.g., "Mermaid flowchart", "PlantUML", "Graphviz DOT"

class FlowDiagramPrompt(Prompt[FlowDiagramPromptInput, str]):
    system_prompt = """
    You are an AI assistant specialized in generating various diagram syntaxes based on descriptions.
    Your goal is to generate clean, valid diagram code for the specified type.
    Provide ONLY the diagram code. Do NOT include any conversational text or markdown code block wrappers.
    
    For Mermaid flowcharts:
    - The code MUST start with 'graph TD' (top-down) or 'graph LR' (left-right).
    - Nodes with custom labels should be defined as `NodeId[Node Label]`.
    - Use correct link label syntax: `A -->|Label| B` or `A --> B : Label`.
    - IMPORTANT: When using bracketed node labels like `NodeA[My Label]`, ensure there is NO colon directly after the closing bracket if it's not part of a link label itself. For example, `NodeA[My Label] --> Target` is correct. Do NOT generate `NodeA[My Label]: Label` or similar. Link labels are placed ON the arrow (`-->|Label|`).
    - All nodes should be defined before or at their first usage.
    
    For PlantUML:
    - The code MUST start with '@startuml' and end with '@enduml'.
    - Use correct syntax for components, actors, use cases, sequences etc.
    
    For Graphviz DOT:
    - The code MUST start with 'digraph G {' or 'graph G {' and end with '}'.
    - Nodes are `node_id [label="Node Label"];`. Edges are `source_id -> target_id;` or `source_id -> target_id [label="Edge Label"];`.
    
    For Mermaid ER Diagrams:
    - The code MUST start with 'erDiagram'.
    
    Ensure all nodes are connected or clearly defined if isolated.
    """
    user_prompt = """
    Generate {{ diagram_type }} diagram code for the following description:
    "{{ description }}"
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.5
    llm_settings: LLMSettings = LLMSettings()

# NEW: Prompt for suggesting data transformations
class SuggestedTransformationPromptInput(BaseModel):
    data_preview: str
    goals: str = ""

class SuggestedTransformationPrompt(Prompt[SuggestedTransformationPromptInput, str]):
    system_prompt = """
    You are a data analyst assistant. Given a DataFrame preview, suggest 3-5 common and useful data transformation operations.
    Focus on operations like filtering, grouping, aggregation, pivoting, merging, joining, cleaning, or feature engineering.
    Provide only a bulleted list of suggestions in natural language. Do NOT provide code or explanations.
    """
    user_prompt = """
    Given the following pandas DataFrame preview:
    ```
    {{ data_preview }}
    ```
    Suggest some useful data transformations that could be applied.
    {% if goals %}
    User's goals: {{ goals }}
    {% endif %}
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.6
    llm_settings: LLMSettings = LLMSettings()

# NEW: Prompt for generating data transformation code with conceptual annotation
class TransformationCodePromptInput(BaseModel):
    data_preview: str # Preview of the DataFrame to be transformed
    transformation_description: str # User's description of desired transformation
    all_df_schemas_json: str # JSON string of all DataFrame schemas for potential merges/joins

class TransformationCodePrompt(Prompt[TransformationCodePromptInput, str]): # REVERTED TO STR
    system_prompt = """
    You are an expert Python data engineer. Your task is to generate Python code for data transformations using polars.
    The original DataFrame `df` will be provided as a **Pandas DataFrame**. Your code MUST perform the following steps:
    1.  Convert the input Pandas DataFrame `df` to a Polars DataFrame.
    2.  Perform all transformations using Polars' efficient API (e.g., `with_columns`, `group_by`, `select`).
    3.  Convert the final transformed Polars DataFrame back to a Pandas DataFrame, and assign it to a variable named `transformed_df`.
    
    Your output MUST be a JSON object with the following keys:
    - `code`: An **array of strings**, where each string is a single line of the Python code. This avoids complex JSON escaping for newlines.
    - `annotation`: A conceptual annotation of the transformation's "shape change" in `(Input Shape) -> (Output Output)` format (e.g., `(rows, cols) -> (new_rows, new_cols)` or `(R, C) -> (R', C')`). Use 'N/A' if not a clear shape change, or provide a symbolic one like `(R, C) -> (R', C')`. This is a conceptual representation, not for direct library validation.
    - `description`: A brief natural language summary of the transformation applied.
    
    Example output JSON for string cleaning/standardization (e.g., phone numbers):
    ```json
    {
        "code": [
            "import polars as pl",
            "import pandas as pd",
            "import re # Import regex module explicitly",
            "import numpy as np",
            "import warnings # Import warnings module",
            "",
            "# Filter out specific warnings that might arise during type conversions or regex on mixed data.",
            "# This is to prevent benign warnings from being interpreted as errors in the Streamlit exec context.",
            "warnings.filterwarnings('ignore', category=FutureWarning)",
            "warnings.filterwarnings('ignore', category=DeprecationWarning)",
            "",
            "# Original df (Pandas DataFrame) is available in scope. Convert it to Polars.",
            "pl_df = pl.DataFrame(df)",
            "",
            "# Robust standardization of 'Phone_Number' column in Polars:",
            "# 1. Ensure column is string type, and fill any nulls with an empty string.",
            "#    Use Polars' `as_str()` and `fill_null('')` for this.",
            "# 2. Use Polars' `str.replace_all` with regex to remove non-digit characters.",
            "#    Polars' regex engine is highly optimized and robust.",
            "pl_df = pl_df.with_columns(",
            "    pl.col('Phone_Number').cast(pl.String).fill_null('')",
            "    .str.replace_all(r'\\\\D', '')", # CORRECTED LINE: Using r'\\\\D' for literal `\D` in a non-raw string",
            "    .alias('Phone_Number')",
            ")",
            "",
            "# Convert the final Polars DataFrame back to Pandas for output.",
            "transformed_df = pl_df.to_pandas()"
        ],
        "annotation": "(R, C) -> (R, C)",
        "description": "Standardized the 'Phone_Number' column by converting to string, filling missing values, and robustly removing all non-digit characters using Polars."
    }
    ```
    
    **CRITICAL GUIDANCE for Data Robustness and Type Handling with Polars (Read Carefully):**
    -   Always `import polars as pl`.
    -   Always `import pandas as pd`.
    -   The input `df` to your generated code will be a **Pandas DataFrame**. Your *first* step must be `pl_df = pl.DataFrame(df)` to convert it to Polars.
    -   Perform *all* data transformations using Polars' API (`pl_df.with_columns(...)`, `pl_df.group_by(...)`, `pl_df.select(...)`, `pl.col(...)`).
    -   Polars' `cast(pl.String)` is equivalent to Pandas' `astype(str)`.
    -   Polars' `fill_null('')` is equivalent to Pandas' `fillna('')`.
    -   Polars' string methods are in the `str` namespace, e.g., `pl.col('column').str.replace_all(r'\\D', '')`. Use double backslashes `\\\\` for regex patterns like `\D` inside Python strings.
    -   **Your *last* step must be to convert the final Polars DataFrame back to Pandas:** `transformed_df = pl_df.to_pandas()`. This ensures compatibility with the Streamlit UI.
    -   Always `import numpy as np` if using `np.nan` or other numpy functions for data handling (though less common in pure Polars, it might be needed for intermediate Pandas conversions or specific numeric types).
    -   Always `import warnings` and use `warnings.filterwarnings` to prevent benign warnings from halting `exec()` execution.
    -   Anticipate and handle `null` values explicitly with Polars' `fill_null`, `drop_nulls`, etc.
    -   Ensure the code is robust, handles common data types, and is efficient. If a merge/join is requested, assume other DataFrames are also converted to Polars (e.g., `other_pl_df = pl.DataFrame(other_df_pandas)`) and then joined with `pl_df.join()`.
    -   If the transformation described is impossible or nonsensical, provide an empty string for `code` and `annotation` and a descriptive `description` of why it's impossible.
    """
    user_prompt = """
    Given the current DataFrame preview (Pandas format):
    ```
    {{ data_preview }}
    ```
    And other available DataFrame schemas (Polars column names):
    {{ all_df_schemas_json }}
    
    Generate Python code (using Polars for transformations) to perform the following:
    "{{ transformation_description }}"
    
    Ensure the final output DataFrame is a Pandas DataFrame assigned to `transformed_df`.
    Provide the output as a JSON object as described in the system prompt.
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.5
    llm_settings: LLMSettings = LLMSettings()

# NEW: Input and Prompt for Wireframe Generation (MOVED HERE from agents.py)
class WireframePromptInput(BaseModel):
    user_description: str
    mukuro_reference: str # The full MukuroL language reference

class WireframePrompt(Prompt[WireframePromptInput, str]):
    system_prompt = """
    You are an AI assistant specializing in generating UI wireframe code using MukuroL, a lightweight markup language.
    Your task is to translate user descriptions into valid and concise MukuroL code.
    
    Adhere strictly to the MukuroL syntax provided in the reference.
    
    Output Format:
    - Provide ONLY the MukuroL code, no conversational text, no markdown code blocks (e.g., no ```mukuro or ```txt).
    - Ensure correct indentation using spaces to represent nesting.
    - The first element on the first line MUST be `page`.
    - Use `gpos` for elements within `grid` containers.
    - For text content that is not an attribute (like 'label' or 'text'), simply write it on an indented line.
    - Strive for a functional and clear wireframe.

    MukuroL Language Reference:
    {{ mukuro_reference }}
    """
    user_prompt = """
    Generate MukuroL code for the following UI wireframe description:
    "{{ user_description }}"
    """
    class LLMSettings(BaseModel):
        temperature: float = 0.8 # Higher temperature for creativity in design
    llm_settings: LLMSettings = LLMSettings()


async def generate_content_with_ragbits_llm(prompt_instance: Prompt) -> str:
    llm = get_ragbits_llm_client()
    try:
        response = await llm.generate(prompt=prompt_instance)
        return response
    except Exception as e:
        return f"Error: An error occurred during content generation: {e}"

def generate_chart_code_with_ragbits(data_preview: str, user_query: str) -> str:
    chart_prompt_input_data = ChartPromptInput(data_preview=data_preview, user_query=user_query)
    chart_prompt_instance = ChartPrompt(chart_prompt_input_data)
    return asyncio.run(generate_content_with_ragbits_llm(chart_prompt_instance))

async def generate_er_diagram_code(description: str) -> str:
    er_prompt_input_data = ERDiagramPromptInput(description=description)
    er_prompt_instance = ERDiagramPrompt(er_prompt_input_data)
    return await generate_content_with_ragbits_llm(er_prompt_instance)

async def generate_flow_diagram_code(description: str, diagram_type: str) -> str:
    flow_prompt_input_data = FlowDiagramPromptInput(description=description, diagram_type=diagram_type)
    flow_prompt_instance = FlowDiagramPrompt(flow_prompt_input_data)
    return await generate_content_with_ragbits_llm(flow_prompt_instance)

async def generate_er_diagram_for_multiple_dfs(description: str, df_schemas_json: str) -> str:
    er_multi_df_prompt_input = ERDiagramMultiDFPromptInput(description=description, df_schemas_json=df_schemas_json)
    er_multi_df_prompt_instance = ERDiagramMultiDFPrompt(er_multi_df_prompt_input)
    return await generate_content_with_ragbits_llm(er_multi_df_prompt_instance)

async def suggest_data_transformations_prompt(data_preview: str) -> str:
    suggest_prompt_input = SuggestedTransformationPromptInput(data_preview=data_preview)
    suggest_prompt_instance = SuggestedTransformationPrompt(suggest_prompt_input)
    return await generate_content_with_ragbits_llm(suggest_prompt_instance)

async def generate_transformation_code_prompt(data_preview: str, transformation_description: str, all_df_schemas_json: str) -> str:
    transform_prompt_input = TransformationCodePromptInput(
        data_preview=data_preview,
        transformation_description=transformation_description,
        all_df_schemas_json=all_df_schemas_json
    )
    transform_prompt_instance = TransformationCodePrompt(transform_prompt_input)
    return await generate_content_with_ragbits_llm(transform_prompt_instance)

# NEW: Function to generate MukuroL wireframe code using the agent
async def generate_mukuro_wireframe_code(user_description: str, mukuro_reference: str, temperature: float = 0.8) -> str:
    llm_client = get_ragbits_llm_client() # Get the initialized LiteLLM client
    
    wireframe_prompt_input = WireframePromptInput(
        user_description=user_description,
        mukuro_reference=mukuro_reference
    )
    wireframe_prompt_instance = WireframePrompt(wireframe_prompt_input)
    wireframe_prompt_instance.llm_settings.temperature = temperature # Apply temperature from UI
    
    try:
        response = await llm_client.generate(prompt=wireframe_prompt_instance)
        return response
    except Exception as e:
        return f"Error: An error occurred during wireframe generation: {e}"
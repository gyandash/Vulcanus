# src/core/code_processor.py
import ast
import json
# Note: The LLM-based generate_mermaid_flow_from_description is now in agents.py
# and the LLM-based data lineage analysis (if needed) will be handled by RagbitsDataLineageAgent.
# This file remains for AST-based code analysis primarily.
# NEW: Constants for common data operations to identify lineage via AST (if still used)
# These patterns can be used by an AST analyzer or by an LLM prompt.
# We keep them here for clarity if AST is to identify these.
DATA_SOURCE_PATTERNS = {
    'pandas.read_csv': 'pandas.read_csv',
    'pandas.read_excel': 'pandas.read_excel',
    'pandas.read_sql': 'pandas.read_sql',
    'boto3.client("s3").get_object': 'boto3.client("s3").get_object',
    'pyspark.sql.DataFrameReader.format': '.read.format',
    'open(..., "r")': 'open(..., "r")',
    'sqlite3.connect': 'sqlite3.connect',
    'neo4j.GraphDatabase.driver': 'neo4j.GraphDatabase.driver',
    'requests.get': 'requests.get',
    'requests.post (as source)': 'requests.post' # HTTP APIs as sources
}
DATA_SINK_PATTERNS = {
    'pandas.DataFrame.to_csv': 'pandas.DataFrame.to_csv',
    'pandas.DataFrame.to_sql': 'pandas.DataFrame.to_sql',
    'boto3.client("s3").put_object': 'boto3.client("s3").put_object',
    'pyspark.sql.DataFrameWriter.format': '.write.format',
    'open(..., "w")': 'open(..., "w")',
    'db_execute_insert': '.execute("INSERT")', # Generic DB insert
    'db_execute_update': '.execute("UPDATE")', # Generic DB update
    'requests.put': 'requests.put', # HTTP APIs as sinks
    'requests.post (as sink)': 'requests.post' # POSTing data
}
# Helper to add parent links to AST nodes (ast.walk doesn't provide them natively)
def ast_with_parents(node, parent=None):
    """Recursively adds a 'parent' attribute to each AST node."""
    node.parent = parent
    for child in ast.iter_child_nodes(node):
        ast_with_parents(child, node)
    return node
def extract_call_chain(node):
    """Extracts a callable chain, e.g., 'pandas.read_csv' from a.b.c()"""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        base = extract_call_chain(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr # Fallback for attributes without clear base (e.g., `df.to_csv`)
    elif isinstance(node, ast.Call): # Handles func() or obj.func() when called directly
        return extract_call_chain(node.func)
    return None
def analyze_python_code_for_flow(code: str) -> dict:
    """
    Analyzes Python code to extract a simplified flow structure (functions and their calls)
    and identifies potential data sources/sinks using AST.
    Returns a dictionary suitable for graph visualization (like JSON Crack or Mermaid).
    """
    flow_data = {
        "nodes": [],
        "edges": [],
        "data_sources_identified": [], # List of identified data sources/inputs labels
        "data_sinks_identified": []    # List of identified data sinks/outputs labels
    }
    nodes_map = {} # Maps node ID to its type (function, class, source, sink, method, global)
    
    # Track which nodes are already added to flow_data["nodes"] to avoid duplicates
    existing_node_ids = set()
    def add_node_if_new(node_id, label, node_type):
        if node_id not in existing_node_ids:
            nodes_map[node_id] = node_type
            flow_data["nodes"].append({"id": node_id, "label": label, "type": node_type})
            existing_node_ids.add(node_id)

    try:
        tree = ast_with_parents(ast.parse(code))
        
        # First pass: identify all functions, classes, and generic scopes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                add_node_if_new(node.name, node.name, "function")
            elif isinstance(node, ast.ClassDef):
                add_node_if_new(node.name, node.name, "class")
        
        # Ensure 'global' node exists for top-level calls
        add_node_if_new("global", "Global Scope", "scope")
        # Second pass: identify calls, create edges, and link to data sources/sinks
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                caller_name = "global" 
                current_parent = node.parent
                while current_parent:
                    if isinstance(current_parent, (ast.FunctionDef, ast.ClassDef)):
                        caller_name = current_parent.name
                        break
                    current_parent = getattr(current_parent, 'parent', None)
                
                # Ensure caller node exists (e.g., if it's a function defined later or implied)
                if caller_name not in nodes_map:
                    if isinstance(current_parent, ast.FunctionDef):
                         add_node_if_new(caller_name, caller_name, "function")
                    elif isinstance(current_parent, ast.ClassDef):
                         add_node_if_new(caller_name, caller_name, "class")
                    else:
                         add_node_if_new(caller_name, caller_name, "scope") # Default if not found explicitly
                full_call_chain = extract_call_chain(node.func)
                callee_id = None
                callee_label = None
                callee_type = "method" # Default for function/method calls
                # Check for data source/sink patterns
                is_data_source = False
                for pattern_name, pattern_str in DATA_SOURCE_PATTERNS.items():
                    if pattern_str in (full_call_chain or ""):
                        callee_id = f"DataSource_{pattern_name}"
                        callee_label = pattern_str
                        callee_type = "data_source"
                        flow_data["data_sources_identified"].append(pattern_str)
                        is_data_source = True
                        break
                
                is_data_sink = False
                if not is_data_source: # Only check for sink if not already identified as source
                    for pattern_name, pattern_str in DATA_SINK_PATTERNS.items():
                        if pattern_str in (full_call_chain or ""):
                            callee_id = f"DataSink_{pattern_name}"
                            callee_label = pattern_str
                            callee_type = "data_sink"
                            flow_data["data_sinks_identified"].append(pattern_str)
                            is_data_sink = True
                            break
                # If not a recognized data source/sink, treat as a regular function/method call
                if not is_data_source and not is_data_sink:
                    if full_call_chain:
                        callee_id = full_call_chain
                        callee_label = full_call_chain.split('.')[-1] # Use last part as label
                        # If it's a simple name (not attr), assume it's a function
                        if isinstance(node.func, ast.Name):
                            callee_type = "function" 
                        else:
                            callee_type = "method" # It's an attribute call
                if callee_id:
                    add_node_if_new(callee_id, callee_label or callee_id, callee_type)
                    # Add edge
                    flow_data["edges"].append({"source": caller_name, "target": callee_id, "label": "calls"})
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")
        return {"nodes": [], "edges": [], "data_sources_identified": [], "data_sinks_identified": []}
    except Exception as e:
        print(f"Error during code analysis: {e}")
        return {"nodes": [], "edges": [], "data_sources_identified": [], "data_sinks_identified": []}
    # Remove duplicate edges
    unique_edges = []
    seen_edges = set()
    for edge in flow_data["edges"]:
        edge_tuple = (edge["source"], edge["target"], edge["label"])
        if edge_tuple not in seen_edges:
            unique_edges.append(edge)
            seen_edges.add(edge_tuple)
    flow_data["edges"] = unique_edges
    return flow_data
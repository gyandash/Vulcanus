# src/core/ragbits_integration.py
import random
import time
import code_ast
from code_ast import ASTVisitor
import tiktoken # For token counting (more accurate for LLM estimates)

def _calculate_loc(code: str) -> int:
    """Calculates non-empty lines of code."""
    return len([line for line in code.splitlines() if line.strip()])

# Helper to map display languages to code-ast supported languages
def _get_code_ast_lang_from_display_lang(display_lang: str) -> str:
    """Converts a display language name to a code-ast (tree-sitter) compatible language identifier."""
    # Based on tree-sitter-languages and common mappings
    lang_map = {
        "python": "python",
        "javascript": "javascript",
        "js": "javascript", # Alias
        "typescript": "javascript", # tree-sitter often treats TS as JS
        "ts": "javascript", # Alias
        "java": "java",
        "c#": "c_sharp", # tree-sitter language name convention
        "csharp": "c_sharp", # Alias
        "go": "go",
        "ruby": "ruby",
        "php": "php",
        "rust": "rust",
        "swift": "swift",
        "c++": "cpp", # tree-sitter language name convention
        "cpp": "cpp", # Alias
        "c": "c",
        "sql": "sql",
        "bash": "bash",
        "sh": "bash", # Alias
        "shell script": "bash", # Alias
        "yaml": "yaml",
        "yml": "yaml", # Alias
        "json": "json",
        "xml": "xml",
        "markdown": "markdown",
        "dockerfile": "dockerfile",
        "hcl": "hcl", # For Terraform HCL
        "html": "html",
        "css": "css",
        # Generic mappings for UI choices
        "general python": "python",
        "general javascript": "javascript",
        "general typescript": "javascript",
        "general java": "java",
        "general c#": "c_sharp",
        "general go": "go",
        "generic yaml": "yaml",
        "generic json": "json",
        "generic xml": "xml",
        "text": "text" # Explicit text type
    }
    
    # Normalize input to lowercase
    normalized_lang = display_lang.lower().replace(" ", "_").replace(".", "")
    return lang_map.get(normalized_lang, "text") # Default to 'text' for unmapped langs

def _calculate_complexity_code_ast(code: str, lang: str) -> float:
    """
    Calculates a simplified complexity score using code-ast.
    Counts control flow, function/class definitions, and calls in the CST.
    """
    code_ast_lang = _get_code_ast_lang_from_display_lang(lang)
    try:
        # code_ast.ast might return None if it cannot parse or language is not compiled/supported
        tree = code_ast.ast(code, lang=code_ast_lang)
        if tree is None:
            # If code-ast returns None (e.g., parsing failed or language not found),
            # fall back to a basic complexity estimation based on LOC.
            # print(f"Warning: code-ast failed for lang='{code_ast_lang}', falling back to LOC-based complexity.")
            return _calculate_loc(code) / 10.0 + 5.0 # Basic complexity for unparsable/unsupported
        
        complexity = 0.0
        
        class ComplexityVisitor(ASTVisitor):
            def __init__(self):
                self.count = 0.0
                self.control_flow_types = {
                    "if_statement", "for_statement", "while_statement", "do_statement",
                    "switch_statement", "try_statement", "catch_clause", "else_clause",
                    "ternary_expression", "case_statement", "default_statement" # common in various languages
                }
                self.definition_types = {
                    "function_definition", "method_definition", "class_definition",
                    "function_declaration", "class_declaration", "struct_declaration",
                    "enum_declaration", "interface_declaration", "variable_declaration", # broad definitions
                }
                self.call_types = {
                    "call_expression", "function_call", "method_invocation", "new_expression" # broad calls
                }
            
            def visit(self, node):
                # Using node.type (string) which represents the grammar rule name
                if node.type in self.definition_types:
                    self.count += 2.0 # More weight for definitions
                elif node.type in self.control_flow_types:
                    self.count += 1.0 # Standard weight for control flow
                elif node.type in self.call_types:
                    self.count += 0.2 # Small weight for each call
                return super().visit(node) # Continue traversal to children
        visitor = ComplexityVisitor()
        tree.visit(visitor)
        return visitor.count
    except Exception as e:
        # Catch any errors during parsing or visiting (e.g., language not found/compiled, grammar issues)
        print(f"Error during code-ast complexity calculation for lang='{lang}' (code_ast_lang='{code_ast_lang}'): {e}")
        # Fallback to a basic estimation if code-ast fails (e.g., `code_ast.ast` raises an error)
        return _calculate_loc(code) / 5.0 + 10.0 # Heuristic if code-ast fails

def get_confidence_score(code: str, lang: str) -> float:
    """
    Provides a confidence score for generated code based on static analysis using code-ast.
    - Parsability by code-ast is a strong indicator.
    - Lower complexity generally leads to higher confidence.
    """
    loc = _calculate_loc(code)
    if loc == 0:
        return 0.0 # No code, no confidence
    try:
        # Attempt to parse to check for syntax validity
        code_ast_lang = _get_code_ast_lang_from_display_lang(lang)
        tree = code_ast.ast(code, lang=code_ast_lang)
        syntax_parsable = (tree is not None)
    except Exception:
        syntax_parsable = False

    base_conf = 0.8 # Start with reasonable base confidence
    if not syntax_parsable:
        base_conf -= 0.5 # Heavily penalize if code-ast cannot parse it (syntax invalid or unsupported lang)
    
    complexity_score = _calculate_complexity_code_ast(code, lang)
    
    # Penalize higher complexity; adjust factor to tune sensitivity
    complexity_penalty = min(0.3, complexity_score * 0.005) # Max 0.3 penalty for very high complexity
    base_conf -= complexity_penalty
    
    # Add a slight bonus for lines of code, indicating more generated content if parsable
    loc_bonus = min(0.1, loc * 0.0005)
    if syntax_parsable:
        base_conf += loc_bonus

    # Add some natural variation for "realism"
    final_confidence = base_conf + random.uniform(-0.05, 0.05)
    
    # Ensure score is within valid bounds [0.0, 1.0]
    return round(max(0.01, min(1.0, final_confidence)), 2)

def get_effort_estimation(code: str, lang: str) -> float:
    """
    Estimates effort (in hours) required to review/refine the generated code using code-ast.
    Based on LOC and a conceptual complexity score. Higher complexity means more effort.
    """
    loc = _calculate_loc(code)
    complexity_score = _calculate_complexity_code_ast(code, lang)
    
    # Base effort per line, plus additional effort for complexity
    effort_per_loc_factor = 0.005 # Less than manual effort per line
    complexity_effort_factor = 0.02 # More effort per complexity point
    
    estimated_effort = (loc * effort_per_loc_factor) + (complexity_score * complexity_effort_factor)
    
    # Ensure a minimum effort even for small, simple code, and some variation
    estimated_effort = max(0.1, estimated_effort) + random.uniform(0.0, 0.2) # Small random offset
    return round(estimated_effort, 2)

def get_original_time_estimate(code: str, lang: str) -> float:
    """
    Estimates the time a human developer would take for the task without AI assistance.
    Assumes a higher base time for manual thought and writing per line.
    Includes complexity for a more realistic estimate.
    """
    loc = _calculate_loc(code)
    complexity_score = _calculate_complexity_code_ast(code, lang) # Factor in complexity for original time
    
    # Higher manual time per line and a base for planning/understanding
    manual_time_per_loc_hours = 0.05
    base_planning_hours = 0.75 # Time to plan/understand before coding
    manual_complexity_factor = 0.05 # Manual handling of complexity is also costly
    
    estimated_original_time = (loc * manual_time_per_loc_hours) + \
                              (complexity_score * manual_complexity_factor) + \
                              base_planning_hours + random.uniform(0.5, 1.5)
    return round(max(0.5, estimated_original_time), 2)

def get_time_saved_estimate(original_time: float, effort_estimation: float) -> float:
    """
    Estimates time saved by using AI.
    Time saved = Original human effort - AI-assisted refinement effort.
    Ensures the time saved is not negative.
    """
    return round(max(0, original_time - effort_estimation), 2)
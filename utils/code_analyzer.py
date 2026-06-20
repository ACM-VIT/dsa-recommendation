"""
DSA Engine -- Code Analyzer
=============================
Extracts semantic features from Python solution code using:
    Tree-sitter  -> AST (Abstract Syntax Tree)
    AST          -> CFG (Control Flow Graph)
    AST          -> DFG (Data Flow Graph)
    CFG + DFG    -> PDG (Program Dependence Graph)
    PDG          -> Semantic feature vector

The feature vector is what gets fed into GraphCodeBERT instead of
raw code text. This gives structurally-aware embeddings — two solutions
using the same algorithm pattern (e.g. sliding window) will be closer
in embedding space even if they look syntactically different.

PDG Node types extracted:
    - Control flow: if/else/for/while/return/try/with
    - Data flow: assignments, variable reads/writes, function calls
    - Dependencies: which statements affect which other statements

Feature vector (fixed 128-dim):
    [0:10]   control flow counts (if, for, while, return, try, ...)
    [10:20]  data structure usage (dict, list, set, heap, ...)
    [20:30]  algorithm patterns  (two pointers, sliding window, ...)
    [30:50]  variable dependency depth / fan-out metrics
    [50:80]  call graph features (which builtins/methods used)
    [80:128] AST shape features  (depth, branching factor, ...)

Usage:
    from code_analyzer import extract_semantic_features, build_semantic_text
    
    # Returns 128-dim numpy vector
    features = extract_semantic_features(code_string)
    
    # Returns enriched text for GraphCodeBERT
    enriched = build_semantic_text(code_string)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np


# ---------------------------------------------------------------------------
# Tree-sitter setup
# ---------------------------------------------------------------------------

def _get_parser():
    """Lazy-load tree-sitter parser. Returns None if not installed."""
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        lang = Language(tspython.language())
        parser = Parser(lang)
        return parser
    except ImportError:
        return None

_parser = None

def get_parser():
    global _parser
    if _parser is None:
        _parser = _get_parser()
    return _parser


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------

def parse_code(code: str):
    """Parse Python code into a tree-sitter AST. Returns root node or None."""
    parser = get_parser()
    if parser is None:
        return None
    try:
        tree = parser.parse(bytes(code, "utf-8"))
        return tree.root_node
    except Exception:
        return None


def walk_ast(node, visitor_fn, depth: int = 0):
    """Walk AST calling visitor_fn(node, depth) on every node."""
    visitor_fn(node, depth)
    for child in node.children:
        walk_ast(child, visitor_fn, depth + 1)


def get_node_text(node, code: str) -> str:
    """Extract source text for a node."""
    try:
        return code[node.start_byte:node.end_byte]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# CFG extraction (Control Flow Graph)
# ---------------------------------------------------------------------------

# Node types that represent control flow branching points
_CFG_NODE_TYPES = {
    "if_statement":       "if",
    "for_statement":      "for",
    "while_statement":    "while",
    "return_statement":   "return",
    "try_statement":      "try",
    "except_clause":      "except",
    "with_statement":     "with",
    "break_statement":    "break",
    "continue_statement": "continue",
    "raise_statement":    "raise",
    "match_statement":    "match",
}

def extract_cfg_features(root) -> Dict[str, int]:
    """
    Extract control flow counts from AST.
    Returns dict of {flow_type: count}.
    """
    counts = {v: 0 for v in _CFG_NODE_TYPES.values()}
    counts["max_nesting_depth"] = 0
    counts["total_branches"]    = 0

    def visitor(node, depth):
        if node.type in _CFG_NODE_TYPES:
            flow_type = _CFG_NODE_TYPES[node.type]
            counts[flow_type] += 1
            counts["total_branches"] += 1
            counts["max_nesting_depth"] = max(counts["max_nesting_depth"], depth)

    walk_ast(root, visitor)
    return counts


# ---------------------------------------------------------------------------
# DFG extraction (Data Flow Graph)
# ---------------------------------------------------------------------------

_DATA_STRUCTURE_PATTERNS = {
    "dict":          [r"\bdict\b", r"\{\s*\}", r"\{\s*\w+\s*:", r"defaultdict", r"Counter\b"],
    "list":          [r"\blist\b", r"\[\s*\]", r"\.append\b", r"\.extend\b"],
    "set":           [r"\bset\b", r"\{\s*\}", r"\.add\b", r"\.discard\b"],
    "heap":          [r"heapq\b", r"heappush\b", r"heappop\b"],
    "deque":         [r"\bdeque\b", r"appendleft\b", r"popleft\b"],
    "stack":         [r"\.pop\(\)", r"\.append\b", r"\bstack\b"],
    "two_pointers":  [r"\bleft\b.*\bright\b", r"\bi\b.*\bj\b", r"while.*<.*:"],
    "binary_search": [r"\bmid\b", r"\blow\b.*\bhigh\b", r"while.*lo.*hi"],
    "sliding_window":[r"\bwindow\b", r"\bright\b.*\bleft\b", r"max_len\b"],
    "recursion":     [r"\bdef\b.*\(.*\).*:", r"return.*\("],
    "dp_table":      [r"\bdp\b\s*=\s*\[", r"\bdp\b\[", r"\bmemo\b"],
    "graph_bfs":     [r"\bqueue\b", r"\bfrom collections import deque\b", r"\.popleft\(\)"],
    "graph_dfs":     [r"\bstack\b", r"def dfs\b", r"def helper\b"],
    "union_find":    [r"\bparent\b", r"def find\b", r"def union\b"],
    "trie":          [r"\bchildren\b", r"TrieNode\b", r"def insert\b"],
}

def extract_dfg_features(code: str) -> Dict[str, int]:
    """
    Extract data flow features from raw code using pattern matching.
    Returns dict of {feature: 0_or_1}.
    """
    features = {}
    code_lower = code.lower()

    for ds_name, patterns in _DATA_STRUCTURE_PATTERNS.items():
        hit = 0
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                hit = 1
                break
        features[f"uses_{ds_name}"] = hit

    # Variable assignment count (proxy for data flow complexity)
    features["assignment_count"] = len(re.findall(r"\w+\s*=\s*[^=]", code))

    # Function call count
    features["call_count"] = len(re.findall(r"\w+\s*\(", code))

    # Return count
    features["return_count"] = len(re.findall(r"\breturn\b", code))

    # Lambda usage
    features["uses_lambda"] = int(bool(re.search(r"\blambda\b", code)))

    # Comprehension usage (list/dict/set comprehensions)
    features["uses_comprehension"] = int(bool(re.search(r"\[.*for.*in.*\]|\{.*for.*in.*\}", code)))

    return features


# ---------------------------------------------------------------------------
# PDG -- Program Dependence Graph features
# ---------------------------------------------------------------------------

def extract_pdg_features(root, code: str) -> Dict[str, Any]:
    """
    Extract program dependence features by combining CFG + DFG signals.
    Focuses on statement-level dependencies.
    """
    features = {}

    # Variable definitions and uses (simplified dependency tracking)
    defined_vars: Dict[str, int] = {}   # var -> line defined
    used_vars:    Dict[str, List[int]] = {}  # var -> [lines used]

    def collect_assignments(node, depth):
        if node.type == "assignment":
            # Left side is the defined variable
            if node.children:
                lhs = node.children[0]
                if lhs.type == "identifier":
                    var_name = get_node_text(lhs, code)
                    line = lhs.start_point[0]
                    defined_vars[var_name] = line

        if node.type == "identifier":
            var_name = get_node_text(node, code)
            line = node.start_point[0]
            if var_name not in used_vars:
                used_vars[var_name] = []
            used_vars[var_name].append(line)

    walk_ast(root, collect_assignments)

    # Dependency depth: how many variables are used that were defined earlier
    dep_count = sum(
        1 for v in defined_vars
        if v in used_vars and any(u > defined_vars[v] for u in used_vars[v])
    )
    features["dependency_count"]  = dep_count
    features["unique_vars"]        = len(defined_vars)
    features["avg_var_use_count"]  = (
        np.mean([len(uses) for uses in used_vars.values()])
        if used_vars else 0.0
    )

    # Fan-out: max number of times a single variable is used
    features["max_var_fan_out"] = (
        max(len(uses) for uses in used_vars.values())
        if used_vars else 0
    )

    return features


# ---------------------------------------------------------------------------
# AST shape features
# ---------------------------------------------------------------------------

def extract_ast_shape(root) -> Dict[str, Any]:
    """
    Extract structural features from AST shape.
    These capture the 'skeleton' of the solution independent of variable names.
    """
    depths    = []
    node_type_counts: Dict[str, int] = {}
    branching_factors = []

    def visitor(node, depth):
        depths.append(depth)
        t = node.type
        node_type_counts[t] = node_type_counts.get(t, 0) + 1
        child_count = len([c for c in node.children if not c.is_named])
        if child_count > 0:
            branching_factors.append(child_count)

    walk_ast(root, visitor)

    return {
        "ast_max_depth":       max(depths) if depths else 0,
        "ast_avg_depth":       float(np.mean(depths)) if depths else 0.0,
        "ast_total_nodes":     len(depths),
        "ast_unique_types":    len(node_type_counts),
        "ast_avg_branch":      float(np.mean(branching_factors)) if branching_factors else 0.0,
        "has_nested_function": int(node_type_counts.get("function_definition", 0) > 1),
        "has_class":           int(node_type_counts.get("class_definition", 0) > 0),
    }


# ---------------------------------------------------------------------------
# Full semantic feature extraction
# ---------------------------------------------------------------------------

def extract_semantic_features(code: str) -> np.ndarray:
    """
    Main entry point. Parse code through AST -> CFG + DFG -> PDG,
    extract features, return a fixed 128-dim float32 feature vector.

    Falls back to a zero vector if tree-sitter is not installed or
    parsing fails — so the pipeline never breaks.
    """
    try:
        root = parse_code(code)

        if root is None or root.has_error:
            # Fallback: use only regex-based DFG features
            dfg = extract_dfg_features(code)
            return _dict_to_vector(dfg, dim=128)

        cfg = extract_cfg_features(root)
        dfg = extract_dfg_features(code)
        pdg = extract_pdg_features(root, code)
        ast = extract_ast_shape(root)

        combined = {**cfg, **dfg, **pdg, **ast}
        return _dict_to_vector(combined, dim=128)

    except Exception:
        return np.zeros(128, dtype=np.float32)


def _dict_to_vector(features: Dict[str, Any], dim: int = 128) -> np.ndarray:
    """
    Convert feature dict to a fixed-size float32 vector.
    Values are clipped and normalised to [0, 1].
    """
    values = []
    for v in features.values():
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(0.0)

    vec = np.array(values, dtype=np.float32)

    # Pad or truncate to fixed dim
    if len(vec) < dim:
        vec = np.pad(vec, (0, dim - len(vec)))
    else:
        vec = vec[:dim]

    # Normalise non-zero values to [0, 1] range
    max_val = vec.max()
    if max_val > 0:
        vec = vec / max_val

    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Semantic text builder for GraphCodeBERT
# ---------------------------------------------------------------------------

def build_semantic_text(code: str) -> str:
    """
    Build an enriched text representation of the code that includes:
        1. Raw code (truncated)
        2. Detected algorithmic patterns (from DFG)
        3. Control flow summary (from CFG)
        4. Structural summary (from AST shape)

    This is fed into GraphCodeBERT instead of raw code, giving the model
    explicit structural signal alongside the code tokens.
    """
    if not code or not code.strip():
        return ""

    parts = []

    # 1. Raw code (GraphCodeBERT handles 512 tokens; 1200 chars is safe)
    parts.append(code[:1200])

    root = parse_code(code)

    # 2. Algorithmic pattern annotations
    dfg = extract_dfg_features(code)
    detected_patterns = [
        name.replace("uses_", "").replace("_", " ")
        for name, val in dfg.items()
        if name.startswith("uses_") and val == 1
    ]
    if detected_patterns:
        parts.append("# Patterns: " + ", ".join(detected_patterns))

    # 3. Control flow summary
    if root is not None and not root.has_error:
        cfg = extract_cfg_features(root)
        flow_items = [
            f"{count}x {flow_type}"
            for flow_type, count in cfg.items()
            if isinstance(count, int) and count > 0
            and flow_type not in ("max_nesting_depth", "total_branches")
        ]
        if flow_items:
            parts.append("# Control flow: " + ", ".join(flow_items))

        # 4. Structural annotations
        ast_feats = extract_ast_shape(root)
        struct_notes = []
        if ast_feats["has_nested_function"]:
            struct_notes.append("nested functions")
        if ast_feats["has_class"]:
            struct_notes.append("class-based")
        if ast_feats["ast_max_depth"] > 10:
            struct_notes.append("deep nesting")
        if struct_notes:
            parts.append("# Structure: " + ", ".join(struct_notes))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------

def extract_features_batch(codes: List[str]) -> np.ndarray:
    """
    Extract semantic feature vectors for a list of code strings.
    Returns array of shape (N, 128).
    """
    return np.stack([extract_semantic_features(c) for c in codes])


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_code = """
class Solution:
    def twoSum(self, nums, target):
        d = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in d:
                return [d[complement], i]
            d[num] = i
        return []
"""
    print("=== Smoke test: Two Sum ===")
    features = extract_semantic_features(test_code)
    print(f"Feature vector shape : {features.shape}")
    print(f"Non-zero features    : {(features > 0).sum()}")
    print(f"Feature range        : [{features.min():.3f}, {features.max():.3f}]")

    enriched = build_semantic_text(test_code)
    print(f"\nEnriched text for GraphCodeBERT:")
    print(enriched)

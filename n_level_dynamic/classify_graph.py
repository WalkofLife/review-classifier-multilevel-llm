"""
Builds a LangGraph state machine to classify reviews into multiple levels
based on a hierarchy CSV and a LLM model loaded via ctransformers.
Supports any number of hierarchy levels.
"""

import re
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain_community.llms import CTransformers
from config import load_config  # You already have this

# ------------------------------
# Setup Config & Logging
# ------------------------------
cfg = load_config()

logging.basicConfig(level=cfg.get("logging").get('level'))
logger = logging.getLogger(__name__)

hierarchy_path = Path(cfg.get("data", {}).get("hierarchy_csv"))
if not hierarchy_path.exists():
    raise FileNotFoundError(f"Required hierarchy CSV not found at {hierarchy_path}. Please check your config.")
hierarchy_df = pd.read_csv(hierarchy_path)

# ------------------------------
# LLM Wrapper (ctransformers only)
# ------------------------------
class LLMWrapper:
    def __init__(self, llm_config:dict):
        self.config = llm_config
        self._init_ctransformers()
        
    def _init_ctransformers(self):
        """Intialize ctransformers LLM from config.yaml settings."""
        try:
            from langchain_community.llms import CTransformers
        except ImportError as e:
            raise RuntimeError(
                "CTransformers not installed. Install it with `pip install ctransformers`."
            ) from e
            
        cfg = self.config.get('ctransformers')
        self.llm = CTransformers(
            model=cfg["model_dir"],
            model_file=cfg.get("model_file"),
            model_type=cfg.get("model_type"),
            config=cfg.get("config")
        )
    
    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

llm = LLMWrapper(cfg.get("llm"))

# ------------------------------
# Agent State
# ------------------------------
class AgentState(TypedDict):
    review: str
    current_level: int
    labels: Dict[str, Optional[str]]

# ------------------------------
# JSON Parser
# ------------------------------
def safe_json_parser(text):
    """
    Try to extract a JSON object from LLM output.

    Parameters:
        text (str): Input text which may contain a JSON object.

    Returns:
        dict or list: Parsed JSON object if found and valid.
        None: If no JSON is found or parsing fails.
    """
    # Search for a JSON-like object { ... } inside the input text.
    # The DOTALL flag allows '.' to match newlines so JSON can span multiple lines.
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        logging.error("No JSON object found.")
        return None
    try:
        # Extract the matched text and parse it into a Python object (dict or list)
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        # Specific error when JSON syntax is invalid
        logging.error(f"Invalid JSON format: {e}")
    except Exception as e:
        # Catch any unexpected errors
        logging.error(f"Unexpected error while parsing JSON: {e}")
    return None


# ------------------------------
# Node Function Factory
# ------------------------------
def make_level_node(level_idx: int, level_name: str, options:List[str], llm: LLMWrapper):
    """Returns a node function that:
    - Classifies the review into one of the 'options' for this hierarchy level.
    - Stores the result in state["labels"].
    - Moves to the next level (increments current_level).
    Also its a function factory (function inside a function to remember parameters)

    Args:
        level_idx (int): Index of the classification level we are at
        level_name (str): _description_
        options (List[str]): Dict storing chosen labels per hierarchy level
        llm (_type_): LLM model
    """
    def node_fn(state: AgentState):
        review = state['review']
        prompt = (
            f"Classify the review into one of these '{level_name}' categories: {', '.join(options)}.\n"
            f"Review: {review}\n"
            "Respond only with a JSON object like {\"label\": \"chosen_label\"}.\n"
        )
        raw = llm.invoke(prompt) # Call llm
        logger.debug(f"[{level_name}] LLM raw response: {raw}")
        parsed_output = safe_json_parser(raw) # Try parsing JSON output
        chosen_label = parsed_output.get("label") if parsed_output else None  # Extract label or None
        state['labels'][level_name] = chosen_label # Store result for this level eg:state['labels'] == {'Main Category': 'Food'}
        state['current_level'] = level_idx + 1
        return state
    return node_fn # Return the customized node function


# ------------------------------
# Build LangGraph
# ------------------------------
def build_hierarchical_graph(hierarchy_df: pd.DataFrame, llm: LLMWrapper):
    levels = list(hierarchy_df.columns)
    sg = StateGraph(AgentState)
    
    def start_node(state: AgentState):
        state['current_level'] = 0
        state['labels'] = {}
        return state
    
    sg.add_node('start_node', start_node)
    
    for idx, level_name in enumerate(levels, start = 1):
        options = sorted(hierarchy_df[level_name].dropna().unique().tolist())
        node_name = f"level_{idx}"
        sg.add_node(node_name, make_level_node(idx, level_name=level_name, options=options, llm=llm))
    
    sg.add_edge('start_node', 'level_1')
    
    for idx in range(1, len(levels) + 1):
        if idx < len(levels):
            sg.add_edge(f"level_{idx}", f"level_{idx+1}")
        else:
            sg.add_edge(f"level_{idx}", END)
    
    sg.set_entry_point('start_node')
    return sg.compile()


# ------------------------------
# Compile and Use Graph
# ------------------------------
graph = build_hierarchical_graph(hierarchy_df, llm)

# Example
initial_state = {
    "review": "The food was too spicy and made me uncomfortable.",
    "current_level": 0,
    "labels": {}
}

final_state = graph.invoke(initial_state)
print(final_state["labels"])
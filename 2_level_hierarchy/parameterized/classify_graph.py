"""
Builds a LangGraph state machine to classify reviews into Level1 and Level2 categories
based on a hierarchy CSV and an LLM model loaded via ctransformers.
"""


from typing import TypedDict
from pathlib import Path
import logging
import pandas as pd

from langgraph.graph import StateGraph, END
from config import load_config

# ------------------------------
# LLM Wrapper (only ctransformers)
# ------------------------------

class LLMWrapper:
    def __init__(self, llm_config: dict):
        """
        Initialize LLM based on the provided config.
        Currently only supports ctransformers.
        """
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
            
        cfg = self.config.get('ctransformers', {})
        self.llm = CTransformers(
            model = cfg["model_dir"],
            model_file = cfg.get('model_file'),
            model_type = cfg.get('model_type'),
            config = cfg.get('config')
        )
        
    def invoke(self, prompt: str) -> str:
        """
        Call the LLM model with a given prompt
        Returns:
            str: Model's output text.
        """
        return self.llm.invoke(prompt)
    

# ------------------------------
# Load config & setup logging
# ------------------------------
cfg = load_config()
logging.basicConfig(level=cfg.get("logging", {}).get("level"))
logger = logging.getLogger(__name__)


hierarchy_path = Path(cfg.get("data", {}).get("hierarchy_csv"))
if not hierarchy_path.exists():
    raise FileNotFoundError(f"Required hierarchy CSV not found at {hierarchy_path}. Please check your config.")
hierarchy_df = pd.read_csv(hierarchy_path)
# ------------------------------
# Instantiate LLM
# ------------------------------
llm = LLMWrapper(cfg.get("llm"))

# ------------------------------
# LangGraph Agent State definition
# ------------------------------
class AgentState(TypedDict):
    review: str
    level1: str
    level2: str

# ------------------------------
# Classification functions
# ------------------------------
def classify_level1(state: AgentState) -> AgentState:
    """Classify review into one of the Level1 categories."""
    review = state["review"]
    level1_options = hierarchy_df["Level1"].dropna().unique().tolist()

    prompt = (
        "Classify the following review into one of these categories:\n"
        f"{', '.join(level1_options)}\n\n"
        f"Review: {review}\n\n"
        "Respond with only the chosen label."
    )
    response = llm.invoke(prompt)
    chosen = response.strip().split("\n")[0] if response else ""
    logger.debug("Level1 response: %s", chosen)
    return {"level1": chosen, **state}

def classify_level2(state: AgentState) -> AgentState:
    """Classify review into one of the Level2 categories based on Level1."""
    review = state["review"]
    level1 = state.get("level1", "")

    if not level1:
        logger.warning("No Level1 chosen; skipping Level2 classification.")
        return {"level2": "", **state}

    level2_options = hierarchy_df[hierarchy_df["Level1"] == level1]["Level2"].dropna().unique().tolist()
    if not level2_options:
        logger.info("No Level2 options for Level1=%s", level1)
        return {"level2": "", **state}

    prompt = (
        f"Parent category is '{level1}'. Classify the review into one of these subcategories:\n"
        f"{', '.join(level2_options)}\n\n"
        f"Review: {review}\n\n"
        "Respond with only the chosen label."
    )
    response = llm.invoke(prompt)
    chosen = response.strip().split("\n")[0] if response else ""
    logger.debug("Level2 response: %s", chosen)
    return {"level2": chosen, **state}

# ------------------------------
# Build LangGraph state machine
# ------------------------------
builder = StateGraph(AgentState)
builder.add_node("level1_node", classify_level1)
builder.add_node("level2_node", classify_level2)
builder.set_entry_point("level1_node")
builder.add_edge("level1_node", "level2_node")
builder.add_edge("level2_node", END)

graph = builder.compile()    
        
        
        








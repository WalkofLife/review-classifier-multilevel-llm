import pandas as pd
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
# Load .env file
load_dotenv()

# Access the token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Use the token to authenticate
from huggingface_hub import login
login(token=hf_token)


# Load hierarchy
# hierarchy_df = pd.read_excel("Category_details.xlsx", sheet_name=  'Hierarchy')
# description_df = pd.read_excel("Category_details.xlsx", sheet_name = 'Description')
# examples_df = pd.read_excel("Category_details.xlsx", sheet_name = 'Examples')
hierarchy_df = pd.read_csv('category_hierarchy.csv')
hierarchy_df.head(2)



from langchain_community.llms import CTransformers
llm = CTransformers(
    model="models",                          # path to folder, not file
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",       # actual file name
    model_type="auto",                      # mandatory for GGUF
    config={"max_new_tokens": 64, "temperature": 0.3}
)

from typing import Dict, List, Optional
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 1. Define the state type
class AgentState(TypedDict):
    review: str # The review text we want to classify
    current_level: int # Index of the classification level we are at
    labels: Dict[str, Optional[str]] # Dict storing chosen labels per hierarchy level, meaning -> Union[str, None]
    
# 2. Helper function: Safe JSON extraction
import re
import json
import logging
logging.basicConfig(level=logging.ERROR)

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
        logging.error("No JSON object found in the provided text.")
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

def make_level_node(level_idx: int, level_name: str, options: List[str], llm):
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
        
        # Build the classification prompt
        prompt = (
            f"Classify the review into one of the these {level_name} categories: {', '.join(options)}.\n"
            f"Review: :{review}\n"
            "Respond only with a JSON object like {\"label\": \"chosen_label\"}.\n"
        )
        
        raw = llm.invoke(prompt) # Call llm
        parsed_output = safe_json_parser(raw) # Try parsing JSON output
        chosen_label = parsed_output.get("label") if parsed_output else None  # Extract label or None
        state['labels'][level_name] = chosen_label # Store result for this level eg:state['labels'] == {'Main Category': 'Food'}
        state['current_label'] = level_idx + 1
        return state
    return node_fn # Return the customized node function

def build_hierarchical_graph(hierarchy_df, llm):
    """Builds Hierarchical graph

    Args:
        heirarchy_df (_type_): DataFrame where each column = hierarchy level
        and values = possible category names for that level.
        llm: LLM object for classification.
    Returns:
        Compiled LangGraph ready to run.
    """
    levels = list(hierarchy_df.columns) # All hierarchy levels
    sg = StateGraph(AgentState) # Create a graph with our state type
    
    # 1) Add start node
    def start_node(state: AgentState):
        state["current_level"] = 0
        state["labels"] = {}
        return state

    sg.add_node("start", start_node)

    # 2) Add all level nodes first (1-based indexing)
    for idx, level_name in enumerate(levels, start=1):   # idx = 1..N
        options = sorted(hierarchy_df[level_name].dropna().unique().tolist())
        node_name = f"level_{idx}"
        sg.add_node(node_name, make_level_node(idx, level_name, options, llm))

    # 3) Now add edges (safe because all nodes exist)
    sg.add_edge("start", "level_1")  # start -> first level

    # Connect level_i -> level_{i+1}, and last level -> END
    for idx in range(1, len(levels) + 1):   # idx = 1..N
        if idx < len(levels):
            sg.add_edge(f"level_{idx}", f"level_{idx+1}")
        else:
            sg.add_edge(f"level_{idx}", END)  # final node -> END


    # 4) Entry Point
    sg.set_entry_point("start")
    # 5) Compile and return the runnable graph
    return sg.compile()
    
        
        
graph = build_hierarchical_graph(hierarchy_df, llm)

initial_state = {
    "review": "The staff was rude and the food was stale.",
    "current_level": 0,
    "labels": {}
}

final_state = graph.invoke(initial_state)
print(final_state["labels"])
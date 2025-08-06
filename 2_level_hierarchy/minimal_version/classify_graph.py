import pandas as pd
# from ctransformers import AutoModelForCausalLM
from langchain_community.llms import CTransformers
from langgraph.graph import StateGraph, END

# Load hierarchy
hierarchy_df = pd.read_csv("category_hierarchy.csv")


import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access the token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Use the token to authenticate
# from huggingface_hub import login
# login(token=hf_token)

from transformers import pipeline

from ctransformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained(
#     "TheBloke/Llama-2-7B-GGUF",
#     model_file="llama-2-7b.Q4_K_M.gguf",
#     model_type="llama",
#     max_new_tokens = 64,
#     temperature = 0.1
# )
# print(model("What is the capital of France?"))

from langchain_community.llms import CTransformers

llm = CTransformers(
    model="models",                          # path to folder, not file
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",       # actual file name
    model_type="auto",                      # mandatory for GGUF
    config={"max_new_tokens": 64, "temperature": 0.3}
)

# print(llm.invoke('what is capital of france?'))

# ---------------- LangGraph Nodes ----------------------
from typing import Annotated, TypedDict # Annotated -> Provides additional context without affecting the type itself, 
from typing import Sequence # To automatically handle the state updates for sequences such as by adding new msg
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The foundataional class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # Reducer function, tells us how to merge new data into current state. Append instead of replace
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    review: str
    level1: str
    level2: str

def classify_level1(state):
    review = state['review']
    level1_options = hierarchy_df['Level1'].unique().tolist()
    prompt = (
        f"Classify the following review into one of these categories: {', '.join(level1_options)}\n"
        f"Review: {review}\n Answer:"
    )
    response = llm.invoke(prompt)
    # return {"level1": response.strip().split("\n")[0], **state}
    # print('Level1 Response' , response)
    return {"level1": response.strip(), **state}

def classify_level2(state):
    review = state['review']
    level1 = state['level1']
    level2_options = hierarchy_df[hierarchy_df['Level1'] == level1]['Level2'].unique().tolist()
    prompt = (
        f"Parent category is '{level1}'. Classify the review into one of these subcategories: {', '.join(level2_options)}\n"
        f"Review: {review}\nAnswer:"
    )
    response = llm.invoke(prompt)
    # print('Level2 Response' , response)
    return {"level2": response.strip().split("\n")[0], **state}


# ---------------- LangGraph Nodes ----------------------
builder = StateGraph(AgentState)
builder.add_node("level1_node", classify_level1)
builder.add_node('level2_node', classify_level2)

builder.set_entry_point('level1_node')
builder.add_edge('level1_node', 'level2_node')
builder.add_edge('level2_node', END)

graph = builder.compile()

This is a **simplified version** of the review classification project.  
It uses a fixed local LLM model via `ctransformers` and directly loads settings from code —  
no config files, no wrapper classes.

'''
minimal_version/
│
├── classify_graph.py       # Core classification logic
├── review_classification.py # Script to run classification on sample reviews
├── category_hierarchy.csv   # Category hierarchy
└── models/                  # Place your GGUF model here
'''

Notes:
* Model path, name, and settings are hardcoded in classify_graph.py.
* This version is great for demos or quick tests, but not recommended for production.
* For a more maintainable, configurable version, see the config_version/ folder.

Process:
1. Install dependencies
pip install -r requirements.txt

3. Download the model
Place your GGUF model file (e.g. mistral-7b-instruct-v0.1.Q4_K_M.gguf)
in the models/ folder.

4. Prepare category hierarchy
Make sure category_hierarchy.csv is in the same folder and contains:
Level1,Level2
Food,Taste
Food,Service
Delivery,Speed

5. run python review_classification.py

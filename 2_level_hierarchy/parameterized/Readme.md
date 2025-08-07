```markdown
# Review Classification (Configurable Version)

This is the **production-ready version** of the review classification project.  
It uses:
- `config.yaml` for all structured, versioned settings  
- `.env` for secrets (e.g., API keys)  
- An **LLM wrapper class** for flexibility in switching models

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install pandas langgraph ctransformers langchain-community pyyaml python-dotenv

### 2. Setup Configuration
Edit config.yaml to set:
llm:
  provider: "ctransformers"
  ctransformers:
    model_dir: "models"
    model_file: "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    model_type: "auto"
    config:
      max_new_tokens: 64
      temperature: 0.3

### 3. Create .env file
HUGGINGFACEHUB_API_TOKEN=your_token_here

### 4.Prepare category hierarchy Make sure category_hierarchy.csv is in the same folder and contains
Level1,Level2
Food,Taste
Food,Service
Delivery,Speed

### 5. Download the model Place your GGUF model file (e.g. mistral-7b-instruct-v0.1.Q4_K_M.gguf) in the 'models/' folder.

### 5. Run classification
run python review_classification.py



🔧 Why This Version?
- Easier to maintain — change models/settings without touching code
- Version-controlled config — track and review changes in Git
- Secure — secrets stay in .env and out of Git
- Scalable — easy to add more models/providers in the future


⚡ Switching Models
To change the model:
1. Download a new GGUF file
2. Update config.yaml: model_file: "new-model.Q4_K_M.gguf"

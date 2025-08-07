"""
Loads project configuration from config.yaml
and environment variables from .env
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file into os.environ, only secret variables
load_dotenv()

# Default config if config.yaml is missing
DEFAULT_CONFIG = {
    "data": {"hierarchy_csv": "category_hierarchy.csv"},
    "llm": {"provider": "ctransformers"},
    "logging": {"level": "INFO"},
}

def load_config(path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file and merge with defaults.
    If YAML file is missing, returns DEFAULT_CONFIG.

    Args:
        path (str): Path to the config.yaml file

    Returns:
        dict: Merged configuration dictionary
    """
    p = Path(path)
    if not p.exists():
        return DEFAULT_CONFIG  # fallback if no config.yaml found

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Shallow merge (YAML overrides defaults)
    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg)
    return merged

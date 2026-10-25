"""
CEM Utils: Config Loader.
Loads YAML configuration files.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Reads a YAML file and returns a nested dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
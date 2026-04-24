"""
ProScale — Configuration Loader
--------------------------------
Loads config/config.yaml and provides typed access to all parameters.
All source files should import from here rather than hardcoding values.

Usage:
    from utils.config import cfg

    lr   = cfg["agent"]["learning_rate"]
    eps  = cfg["agent"]["episodes"]          # training episodes
    prob = cfg["environment"]["arrival_prob_burst"]
"""

import yaml
from pathlib import Path
from typing import Optional

# Resolve config path relative to project root regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Optional[str] = None) -> dict:
    """Load and return the ProScale configuration as a nested dict."""
    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Module-level singleton — import cfg directly for convenience
cfg = load_config()

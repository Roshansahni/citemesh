"""Load config.yaml + .env. Environment variables override the YAML."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["provider"] = os.getenv("LLM_PROVIDER", cfg["provider"]).lower()
    if os.getenv("OLLAMA_HOST"):
        cfg["ollama"]["host"] = os.environ["OLLAMA_HOST"]
    r = cfg["research"]
    for key in ("local_pdf_dir", "cache_file", "output_dir"):
        r[key] = str(ROOT / r[key])
    return cfg

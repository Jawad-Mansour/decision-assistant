"""Path resolution for both local and Docker environments."""

import os
from pathlib import Path


def get_chroma_path() -> Path:
    """Return the correct Chroma DB path for current environment."""
    
    # 1. Check environment variable first (Docker or explicit override)
    env_path = os.environ.get("CHROMA_PERSIST_DIRECTORY")
    if env_path:
        return Path(env_path)
    
    # 2. Check if we're in Docker (has /app directory with data)
    if Path("/app/data/chroma_db").exists() or (Path("/app").exists() and Path("/app/data").exists()):
        return Path("/app/data/chroma_db")
    
    # 3. Local development - from project root
    # backend/app/core/paths.py -> backend/ -> project_root/
    project_root = Path(__file__).resolve().parents[3]
    chroma_path = project_root / "data" / "chroma_db"
    
    # Create if doesn't exist (for first-time setup)
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    return chroma_path


def get_models_path() -> Path:
    """Return the correct models path."""
    
    env_path = os.environ.get("MODELS_DIRECTORY")
    if env_path:
        return Path(env_path)
    
    if Path("/app/models").exists():
        return Path("/app/models")
    
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "models"
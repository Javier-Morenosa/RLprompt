"""Shared paths for Human-Watch demo. State files live in data/."""

from pathlib import Path

# Raíz del proyecto RLprompt
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Carpeta de archivos de estado (.md, .json)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

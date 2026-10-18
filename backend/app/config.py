from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "ml" / "models"
DATASET_DIR = ROOT_DIR / "ml" / "datasets"
REPORT_DIR = ROOT_DIR / "ml" / "reports"
DATABASE_PATH = ROOT_DIR / "backend" / "data" / "history.sqlite3"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
HISTORY_ENABLED = os.getenv("HISTORY_ENABLED", "true").lower() == "true"

for directory in (MODEL_DIR, DATASET_DIR, REPORT_DIR, DATABASE_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)
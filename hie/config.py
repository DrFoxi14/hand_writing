from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.getenv("HIE_HOME", Path.cwd())).resolve()
DATASET = ROOT / "dataset"
RAW = DATASET / "raw"
PROCESSED = DATASET / "processed"
ANNOTATIONS = DATASET / "annotations"
EVALUATION = DATASET / "evaluation"
MODELS = ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
PROFILES = MODELS / "writer_profiles"
DB_PATH = DATASET / "hie.sqlite3"

for _path in (RAW, PROCESSED, ANNOTATIONS, EVALUATION, CHECKPOINTS, PROFILES):
    _path.mkdir(parents=True, exist_ok=True)

from __future__ import annotations
import hashlib
import json
import mimetypes
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hie.config import DB_PATH, RAW, PROCESSED, ANNOTATIONS

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

class DatasetStore:
    """Append-only source store. Original uploads are never modified or overwritten."""
    def __init__(self, db_path: Path | str = DB_PATH, raw_dir: Path | str = RAW):
        self.db_path = Path(db_path)
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS pages (
                id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE, original_name TEXT NOT NULL,
                raw_path TEXT NOT NULL, processed_path TEXT, width INTEGER, height INTEGER,
                created_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS line_annotations (
                id TEXT PRIMARY KEY, page_id TEXT NOT NULL, x INTEGER NOT NULL, y INTEGER NOT NULL,
                width INTEGER NOT NULL, height INTEGER NOT NULL, text TEXT NOT NULL,
                crop_path TEXT NOT NULL, split TEXT NOT NULL DEFAULT 'train', created_at TEXT NOT NULL,
                FOREIGN KEY(page_id) REFERENCES pages(id)
            );
            CREATE TABLE IF NOT EXISTS glyphs (
                id TEXT PRIMARY KEY, character TEXT NOT NULL, position TEXT NOT NULL,
                image_path TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
                source_page_id TEXT, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_lines_page ON line_annotations(page_id);
            CREATE INDEX IF NOT EXISTS idx_glyphs_char_pos ON glyphs(character, position);
            """)

    def ingest(self, source: bytes, filename: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {ext or '(none)'}")
        if not source:
            raise ValueError("The uploaded image is empty")
        digest = hashlib.sha256(source).hexdigest()
        with self._connect() as conn:
            existing = conn.execute("SELECT * FROM pages WHERE sha256=?", (digest,)).fetchone()
            if existing:
                return {**dict(existing), "duplicate": True}
        page_id = uuid.uuid4().hex
        safe_name = f"{page_id}_{digest[:10]}{ext}"
        dest = self.raw_dir / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects existing raw data against accidental overwrite.
        with dest.open("xb") as fh:
            fh.write(source)
        try:
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(source)) as im:
                width, height = im.size
        except Exception:
            dest.unlink(missing_ok=True)
            raise ValueError("The upload is not a readable image")
        created = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pages(id,sha256,original_name,raw_path,width,height,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                (page_id, digest, Path(filename).name, str(dest), width, height, created, json.dumps(metadata or {})),
            )
        return {"id": page_id, "sha256": digest, "original_name": Path(filename).name,
                "raw_path": str(dest), "width": width, "height": height, "created_at": created,
                "metadata_json": json.dumps(metadata or {}), "duplicate": False}

    def get_page(self, page_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pages WHERE id=?", (page_id,)).fetchone()
        return dict(row) if row else None

    def list_pages(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM pages ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def set_processed(self, page_id: str, path: Path, metadata: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE pages SET processed_path=?, metadata_json=? WHERE id=?",
                         (str(path), json.dumps(metadata), page_id))

    def add_line_annotation(self, page_id: str, bbox: tuple[int, int, int, int], text: str,
                             crop_path: Path, split: str = "train") -> dict[str, Any]:
        if not self.get_page(page_id):
            raise KeyError(f"Unknown page id: {page_id}")
        x, y, w, h = map(int, bbox)
        if min(x, y, w, h) < 0 or w < 2 or h < 2:
            raise ValueError("Bounding box must have positive width and height")
        if not text.strip():
            raise ValueError("Transcription cannot be empty")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        with self._connect() as conn:
            existing_splits = {row[0] for row in conn.execute(
                "SELECT DISTINCT split FROM line_annotations WHERE page_id=?", (page_id,)
            ).fetchall()}
        if existing_splits and existing_splits != {split}:
            raise ValueError("All line crops from a page must use the same split; keep validation pages entirely held out")
        ann_id = uuid.uuid4().hex
        created = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("INSERT INTO line_annotations VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (ann_id, page_id, x, y, w, h, text, str(crop_path), split, created))
        return {"id": ann_id, "page_id": page_id, "bbox": [x,y,w,h], "text": text,
                "crop_path": str(crop_path), "split": split, "created_at": created}

    def list_line_annotations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM line_annotations ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def add_glyph(self, character: str, position: str, image_path: Path,
                  verified: bool = False, source_page_id: str | None = None) -> dict[str, Any]:
        if len(character) != 1:
            raise ValueError("A glyph record must contain exactly one Unicode code point")
        if position not in {"initial", "medial", "final", "isolated", "any"}:
            raise ValueError("Invalid glyph position")
        glyph_id = uuid.uuid4().hex
        created = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("INSERT INTO glyphs VALUES(?,?,?,?,?,?,?)",
                         (glyph_id, character, position, str(image_path), int(verified), source_page_id, created))
        return {"id": glyph_id, "character": character, "position": position,
                "image_path": str(image_path), "verified": verified, "source_page_id": source_page_id}

    def list_glyphs(self, verified_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM glyphs" + (" WHERE verified=1" if verified_only else "")
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY character,position,created_at").fetchall()
        return [dict(row) for row in rows]

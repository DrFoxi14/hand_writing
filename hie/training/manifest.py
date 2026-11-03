from __future__ import annotations
import json
from pathlib import Path
from hie.dataset.store import DatasetStore
from hie.config import ANNOTATIONS


def export_line_manifest(store: DatasetStore | None = None, target: Path | str = ANNOTATIONS / "lines.jsonl") -> Path:
    store = store or DatasetStore()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = store.list_line_annotations()
    payload = [{"id":r["id"],"page_id":r["page_id"],"image":str(Path(r["crop_path"]).resolve()),
                "text":r["text"],"split":r["split"]} for r in rows]
    target.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in payload)+("\n" if payload else ""),encoding="utf-8")
    return target

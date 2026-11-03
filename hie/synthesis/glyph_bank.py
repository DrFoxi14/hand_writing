from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Iterable
import random
import numpy as np
from PIL import Image
from hie.dataset.store import DatasetStore
from hie.config import ANNOTATIONS

class InsufficientGlyphCoverage(ValueError):
    def __init__(self, missing: list[str], positions: dict[str, list[str]] | None = None):
        self.missing = missing
        self.positions = positions or {}
        pretty = ", ".join(repr(ch) for ch in missing)
        super().__init__(f"Missing verified handwriting samples for: {pretty}. Add verified glyph samples; HIE will not invent them.")

class GlyphBank:
    def __init__(self, store: DatasetStore | None = None):
        self.store = store or DatasetStore()

    def variants(self) -> dict[tuple[str,str], list[dict]]:
        grouped: dict[tuple[str,str], list[dict]] = defaultdict(list)
        for glyph in self.store.list_glyphs(verified_only=True):
            if Path(glyph["image_path"]).exists():
                grouped[(glyph["character"], glyph["position"])].append(glyph)
        return grouped

    @staticmethod
    def position_for(text: str, index: int) -> str:
        prev_space = index == 0 or text[index - 1].isspace()
        next_space = (index == len(text) - 1 or text[index + 1].isspace()
                      or text[index + 1] in ",.;:!?)]}”’\"")
        if prev_space and next_space:
            return "isolated"
        if prev_space:
            return "initial"
        if next_space:
            return "final"
        return "medial"

    def contextual_gaps(self, text: str) -> dict[str, list[str]]:
        """Return character positions that lack a verified exact-position or `any` sample."""
        grouped = self.variants()
        required: dict[str, set[str]] = defaultdict(set)
        for index, ch in enumerate(text):
            if not ch.isspace():
                required[ch].add(self.position_for(text, index))
        gaps: dict[str, list[str]] = {}
        for ch, positions in required.items():
            if grouped.get((ch, "any")):
                continue
            missing = sorted(pos for pos in positions if not grouped.get((ch, pos)))
            if missing:
                gaps[ch] = missing
        return gaps

    def missing(self, text: str) -> list[str]:
        return sorted(self.contextual_gaps(text))

    def coverage_report(self, text: str) -> dict:
        chars = sorted({ch for ch in text if not ch.isspace()})
        gaps = self.contextual_gaps(text)
        available = [ch for ch in chars if ch not in gaps]
        missing = sorted(gaps)
        return {"required": chars, "available": available, "missing": missing,
                "missing_positions": gaps,
                "coverage": len(available) / max(1, len(chars))}

    def add_from_array(self, character: str, position: str, rgba: np.ndarray,
                       verified: bool = False, source_page_id: str | None = None) -> dict:
        if len(character) != 1:
            raise ValueError("character must be exactly one Unicode code point")
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("glyph image must be RGBA")
        glyph_dir = ANNOTATIONS / "glyphs"
        glyph_dir.mkdir(parents=True, exist_ok=True)
        import uuid
        path = glyph_dir / f"{uuid.uuid4().hex}.png"
        Image.fromarray(rgba.astype(np.uint8), "RGBA").save(path)
        return self.store.add_glyph(character, position, path, verified, source_page_id)

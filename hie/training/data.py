from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LineDataset(Dataset):
    def __init__(self, manifest: Path | str, split: str = "train", height: int = 64,
                 pixels_per_char: int = 20, vocab: dict[str,int] | None = None):
        self.manifest = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = [r for r in self.rows if r.get("split", "train") == split and Path(r["image"]).exists()]
        self.height, self.pixels_per_char = height, pixels_per_char
        self.vocab = vocab or {ch:i+1 for i,ch in enumerate(sorted({c for r in self.rows for c in r["text"]}))}
        if not self.rows: raise ValueError(f"No labeled line images found for split={split!r}")
        unknown = sorted({ch for r in self.rows for ch in r["text"] if ch not in self.vocab})
        if unknown: raise ValueError(f"Vocabulary is missing characters: {unknown}")

    def __len__(self): return len(self.rows)

    def _image(self, path: str, width: int) -> torch.Tensor:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None: raise ValueError(f"Unable to read line crop: {path}")
        # White paper -> 0, ink -> 1. This normalizes different photograph exposures.
        gray = cv2.resize(gray, (width, self.height), interpolation=cv2.INTER_AREA)
        ink = (255.0-gray.astype(np.float32))/255.0
        return torch.from_numpy(ink[None,:,:])

    def __getitem__(self, index):
        row = self.rows[index]
        text = row["text"]
        ids = torch.tensor([self.vocab[c] for c in text], dtype=torch.long)
        width = max(16, len(text)*self.pixels_per_char)
        target = self._image(row["image"], width)
        style_row = self.rows[(index+1) % len(self.rows)]
        style = self._image(style_row["image"], 256)
        return {"chars": ids, "target": target, "style": style, "text": text}

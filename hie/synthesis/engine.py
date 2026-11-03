from __future__ import annotations
import random
import numpy as np
from PIL import Image
from hie.synthesis.glyph_bank import GlyphBank, InsufficientGlyphCoverage


class GlyphComposer:
    """Verified-sample compositor. It never substitutes a generic font for a missing glyph."""

    def __init__(self, bank: GlyphBank | None = None):
        self.bank = bank or GlyphBank()

    def _select(self, ch: str, position: str, grouped: dict, rng: random.Random) -> Image.Image:
        options = grouped.get((ch, position)) or grouped.get((ch, "any"))
        if not options:
            raise InsufficientGlyphCoverage([ch])
        selected = rng.choice(options)
        return Image.open(selected["image_path"]).convert("RGBA")

    def render(self, text: str, max_width: int = 1200, line_spacing: float = 1.65,
               char_spacing: float = 0.10, seed: int | None = None,
               variation: float = 0.015) -> Image.Image:
        if not text:
            raise ValueError("Text cannot be empty")
        missing = self.bank.missing(text)
        if missing:
            raise InsufficientGlyphCoverage(missing)

        grouped = self.bank.variants()
        rng = random.Random(seed)
        # Create each glyph from a verified handwritten crop, retaining its source context.
        glyphs: dict[int, Image.Image] = {}
        for i, ch in enumerate(text):
            if not ch.isspace():
                glyphs[i] = self._select(ch, self.bank.position_for(text, i), grouped, rng)
        target_h = max(1, int(np.median([im.height for im in glyphs.values()])))
        letter_gap = max(0, int(round(target_h * char_spacing)))
        word_gap = max(4, int(round(target_h * 0.42)))
        line_height = max(target_h, int(round(target_h * line_spacing)))
        available_width = max_width - 8
        if available_width < 16:
            raise ValueError("max_width is too small for handwriting rendering")

        # Each output line is a list of (glyph, x, y). Wraps happen at whitespace.
        output_lines: list[list[tuple[Image.Image, int, int]]] = []
        widest_line = 1
        global_offset = 0
        for paragraph in text.split("\n"):
            current: list[tuple[Image.Image, int, int]] = []
            x = 0
            i = 0
            pending_spaces = 0
            while i < len(paragraph):
                if paragraph[i].isspace():
                    pending_spaces += 1
                    i += 1
                    continue
                start = i
                while i < len(paragraph) and not paragraph[i].isspace():
                    i += 1
                word_indices = list(range(start, i))
                word_items: list[tuple[Image.Image, int, int]] = []
                word_width = 0
                for j, local_index in enumerate(word_indices):
                    original_index = global_offset + local_index
                    source = glyphs[original_index]
                    scale = 1.0 + rng.uniform(-variation, variation) if variation else 1.0
                    width = max(1, int(round(source.width * scale)))
                    height = max(1, int(round(source.height * scale)))
                    glyph = source.resize((width, height), Image.Resampling.LANCZOS)
                    gy = max(0, (target_h - height) // 2)
                    gx = word_width + (letter_gap if j else 0)
                    word_items.append((glyph, gx, gy))
                    word_width = gx + width
                gap = (pending_spaces * word_gap) if x else 0
                if x and x + gap + word_width > available_width:
                    widest_line = max(widest_line, x)
                    output_lines.append(current)
                    current, x, gap = [], 0, 0
                if word_width > available_width:
                    raise ValueError("A single word exceeds max_width; increase the writing area")
                x += gap
                for glyph, gx, gy in word_items:
                    current.append((glyph, x + gx, gy))
                x += word_width
                pending_spaces = 0
            widest_line = max(widest_line, x)
            output_lines.append(current)
            global_offset += len(paragraph) + 1

        canvas_width = max(16, min(max_width, widest_line + 8))
        canvas_height = max(target_h, len(output_lines) * line_height) + 8
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
        for row, items in enumerate(output_lines):
            y0 = 4 + row * line_height
            for glyph, x, y in items:
                if x + glyph.width > canvas_width:
                    raise RuntimeError("Layout width invariant failed")
                canvas.alpha_composite(glyph, (x + 4, y0 + y))
        return canvas

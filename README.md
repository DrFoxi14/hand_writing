# Handwriting Identity Engine (HIE)

A local-first research workbench for learning the visual characteristics of one writer's handwriting from original notebook photographs. HIE prioritizes source preservation, measurable image processing, verified training annotations, explicit character coverage, and minimally invasive notebook compositing.

> **Current status: research prototype, not a finished high-fidelity handwriting clone.** It ingests and preprocesses notebook photos, builds a provisional writer profile, stores verified line/glyph annotations, offers a strict sample-based synthesis baseline, includes a writer-conditioned PyTorch model/training path, and composites RGBA overlays only inside an approved polygon. A small number of untranscribed pages is not sufficient to train or validate a reliable arbitrary-text model; HIE will report missing glyphs rather than invent them.

## What is implemented

- Append-only page ingestion with SHA-256 duplicate detection. Original uploads are stored under `dataset/raw/` and never edited.
- Page quadrilateral detection with a full-frame fallback, perspective rectification, blue/burgundy ink extraction, grid-spacing estimates, and approximate text-line proposals.
- A provisional geometry/ink profile. Component statistics are explicitly labelled approximate because connected cursive strokes are not individual characters.
- SQLite-backed line and glyph annotation records, crop export, and a JSONL training manifest.
- Verified-glyph synthesis with initial/medial/final/isolated/any variants, controlled micro-variation, and a hard contextual-coverage gate. A character is not marked covered unless every position required by the requested text has a verified sample (or an `any` sample). It uses real cropped handwriting samples, not a generic handwriting font.
- A PyTorch writer-conditioned baseline: character sequence encoder + reference-line style encoder + convolutional image decoder.
- Pixel-preserving compositing: every pixel outside the user-specified polygon is copied byte-for-byte from the original RGB image. A four-corner polygon also perspective-warps the overlay to match the selected writing plane; a 3+ point mask can be used without a perspective warp.
- FastAPI local interface for dataset management, annotation, profile inspection, synthesis, training readiness, and reconstruction.
- Hardware detection for NVIDIA CUDA, Apple Silicon MPS, and CPU fallback.

## Not implemented / limitations

- No trained checkpoint is included. Training must be performed after line crops have been manually transcribed.
- Automatic page boundary detection is conservative and may fall back to the full camera frame. Curved-page dewarping is not yet implemented; projective correction cannot flatten arbitrary book curvature.
- The sample-based renderer is a verified glyph collage baseline. It does not yet reproduce continuous pen trajectories or reliably join cursive letter connections. Do not confuse this with neural handwriting synthesis.
- The current PyTorch model is a first supervised baseline, not a validated state-of-the-art implementation. Its training threshold of 40 line crops is a guardrail against obviously tiny datasets, not a claim that 40 lines are enough for good generalization.
- OCR/transcription is manual in this version. This avoids silently accepting English-only OCR errors on Romanian diacritics and mathematical notation.
- The browser interface currently uses rectangular crops for annotations and polygon masks for compositing; complex per-word replacement and inpainting remain future work.

## Requirements

- Python 3.10+
- macOS Apple Silicon, NVIDIA CUDA, or CPU. The pipeline works on CPU; training is substantially slower without acceleration.
- `pip` or `uv`

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e '.[training,test]'
```

If you only need dataset analysis, API, and annotation, install `pip install -e '.[test]'` and omit the training extra.

## Run the local application

```bash
hie-server
```

Open `http://127.0.0.1:8000`. The server binds to loopback by default. Keep it on loopback because handwriting images are personal biometric-like data. Do not expose the service to a public network without adding authentication and TLS.

To override the data root:

```bash
HIE_HOME=/absolute/path/to/private-hie-data hie-server
```

The data root contains raw images, processed derivatives, SQLite records, and local checkpoints. It is not intended for Git commits.

## Suggested first workflow

1. Open **Dataset** and upload your original notebook photographs. The originals remain unchanged; rectified pages and ink masks are saved separately.
2. Open **Handwriting profile** and inspect page geometry, line counts, approximate component dimensions, slant, ink density, and median ink colour.
3. Open **Annotation**. Draw a box around each complete line, enter its exact transcription, and save it. Use a held-out validation split for some pages; do not put every crop into training.
4. Add individual glyph crops only when the crop contains exactly one character. Mark each glyph verified only after visual checking. Include contextual forms: initial, medial, final, and isolated.
5. In **Generate**, check coverage before requesting output. Missing characters are reported explicitly. HIE will not substitute a system font.
6. For notebook compositing, select a polygon limited to the intended writing area. Only that polygon may be changed.
7. Add more varied samples before attempting model training. The model training script refuses fewer than 40 labelled training lines by default; a credible personal model will generally need substantially more, plus held-out pages and careful coverage of Romanian diacritics, numbers, punctuation, and mathematical notation.

## Training

Annotations are exported to `dataset/annotations/lines.jsonl`. To train the first writer-conditioned baseline:

```bash
hie-train --manifest dataset/annotations/lines.jsonl \\
  --output models/checkpoints/writer_v1.pt \\
  --epochs 30 --min-lines 40
```

The training data must contain records with an image crop path, exact transcription, and a `split` value (`train` or `validation`). The API enforces one split per page; keep complete pages/sessions separate to avoid measuring memorization of the same notebook page. Training records use a white-paper/black-ink target mask so the graph grid is not learned as handwriting. Checkpoints and metrics are ignored by Git.

## API overview

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | API and accelerator status |
| `POST /api/pages` | Import a notebook photograph |
| `GET /api/pages` | List source pages |
| `GET /api/pages/{id}/image?kind=rectified\\|mask\\|original` | View source or derivative |
| `GET /api/profile` | Compute provisional handwriting metrics |
| `POST /api/annotations/lines` | Save a transcribed line crop |
| `POST /api/annotations/glyphs` | Save a character crop |
| `GET /api/coverage?text=...` | Check verified glyph coverage |
| `POST /api/generate` | Compose a line from verified glyph crops |
| `POST /api/reconstruct` | Composite an RGBA overlay inside an authorized polygon |
| `POST /api/generate/neural` | Generate from a trained writer-conditioned checkpoint |
| `POST /api/evaluate` | Compute aligned image and transcription metrics |
| `GET /api/training/status` | Inspect annotation and training readiness |
| `POST /api/training/run` | Train the local writer-conditioned checkpoint |

## Tests

```bash
pytest
```

Tests are local and synthetic; they do not require private notebook photographs. No claim is made about handwriting similarity until held-out evaluation has been run on newly captured samples.

## Privacy and repository hygiene

- Never commit source notebook photographs, crops, annotation manifests, SQLite databases, writer profiles, or checkpoints.
- `dataset/`, local checkpoints, and profile outputs are covered by `.gitignore`.
- Run locally and keep the API bound to `127.0.0.1`.
- Use only handwriting samples you own or have explicit permission to process. Do not use generated handwriting to impersonate another person or create misleading signatures, official records, or evidence.

## Architecture and research choices

See [`docs/architecture.md`](docs/architecture.md), [`docs/research.md`](docs/research.md), and [`docs/evaluation.md`](docs/evaluation.md).

## License

MIT for this repository's original code. Third-party model code, weights, and datasets retain their own licenses; review them independently before integration.

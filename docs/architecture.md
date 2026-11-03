# Architecture

## Data flow

```text
Original notebook photo
  └─ SHA-256 + append-only raw store
       └─ page quadrilateral estimate / perspective rectification
            ├─ rectified colour page
            ├─ blue + burgundy ink mask
            ├─ grid-spacing estimate
            └─ approximate line-box proposals
                 ├─ manual line transcription → line crops → JSONL manifest → PyTorch training
                 └─ manual character crops → verified glyph bank → coverage-gated sample renderer

Generated RGBA overlay + explicit polygon
  └─ alpha compositing inside polygon only
       └─ unchanged original pixels everywhere outside polygon
```

## Module responsibilities

- `hie/dataset/store.py`: append-only page store, SHA-256 deduplication, SQLite metadata, annotation records.
- `hie/vision/preprocess.py`: conservative page geometry, warp, ink mask, grid-spacing estimates, line proposals.
- `hie/vision/profile.py`: provisional writer profile from rectified images and ink masks.
- `hie/synthesis/glyph_bank.py`: stores verified contextual glyph examples and computes coverage.
- `hie/synthesis/engine.py`: composes only verified crops; missing characters are fatal, not approximated.
- `hie/training/model.py`: reference-line style encoder + Unicode character-sequence encoder + image decoder.
- `hie/training/data.py`: reads manually transcribed line crops; keeps text labels in Unicode.
- `hie/reconstruction/composite.py`: strict alpha compositing, polygon mask, reverse perspective-warp utility.
- `hie/api/main.py`: local FastAPI service and annotation/generation endpoints.

## Data contracts

### Page record

A page has a UUID, SHA-256 digest, original filename, immutable raw path, optional processed path, original width/height, creation time, and preprocessing metadata. Exact duplicate bytes return the prior record rather than writing another raw file.

### Line annotation

`{id, page_id, bbox:[x,y,w,h], text, crop_path, split, created_at}`

The `text` string is exact Unicode. Romanian diacritics are code points, not approximated ASCII equivalents. `split` is `train` or `validation`; the store enforces one split per complete page to prevent crop-level leakage.

### Glyph record

`{id, character, position, image_path, verified, source_page_id, created_at}`

`position ∈ {initial, medial, final, isolated, any}`. Unverified samples are excluded from generation.

### Training manifest

JSONL records: `{"id":"...","page_id":"...","image":"/absolute/path/to/crop.png","text":"...","split":"train"}`.

## Notebook compositing invariant

The original image is the immutable background. The renderer accepts a same-size RGBA overlay and a polygon in original-photo pixel coordinates. A four-corner polygon is ordered and used as a perspective target; other polygons are mask-only. The polygon defines the only authorized region. For all pixels outside the polygon, `output[y,x] == original[y,x]` is asserted before returning. Perspective transform utilities can map transparent overlays between rectified and camera coordinates, but automatic curvature flattening is not part of this release.

## Training model

The first neural baseline is a sequence-to-image generator:

1. A Unicode character embedding and bidirectional GRU encode the input text.
2. A CNN encodes a reference handwriting line into a writer-style vector.
3. Both representations are expanded into a spatial feature map.
4. A convolutional decoder predicts an ink-probability image.

This is a starting experiment. It is not yet adversarially trained, does not yet model explicit pen trajectories, and has no measured similarity result. It should be compared against FW-GAN and GANwriting once the writer-specific line dataset is large enough.

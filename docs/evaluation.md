# Evaluation plan

Do not describe the output as high-fidelity until this protocol has been completed and results are recorded.

## Dataset split

- Split by full notebook page or writing session, not random crops from the same page.
- Keep at least one complete page/session held out from training.
- Include unseen word combinations and held-out sentences with Romanian diacritics, punctuation, numbers, and mathematical expressions.
- Preserve a manifest and hashes for every evaluation crop.

## Metrics to implement/track

| Metric | Method | Interpretation |
|---|---|---|
| Character transcription accuracy | CER and WER against exact reference text | Content correctness; not visual similarity. |
| Character shape similarity | Aligned crop SSIM / normalized contour distance, grouped by code point and position | Glyph shape fidelity. |
| Proportion error | Height/width distribution distance by code point | Size and proportion consistency. |
| Stroke thickness | Skeleton-distance / distance-transform width statistics | Pen appearance. |
| Slant consistency | Robust angle distribution difference | Inclination stability. |
| Spacing | Inter-character and inter-word gap distributions | Rhythm and density. |
| Baseline alignment | Baseline residual in local line coordinates | Vertical writing regularity. |
| Writer identity similarity | A writer-discriminative embedding trained/evaluated on held-out writers and samples | Whether style embedding identifies the same writer. Avoid using a generic embedding as sole proof. |
| Grid alignment | Ink-to-grid line residual in rectified space | Notebook geometry fit. |
| Pixel preservation | Exact equality of output and input outside the authorized mask | Reconstruction safety invariant; tested in this version. |

## Visual review

For each held-out line, store original, generated, aligned overlay, and absolute-difference image. Review by character position and class. Separate failures caused by transcription, missing coverage, generation, layout, and compositing.

## Reporting template

Report the dataset size, held-out page count, unique character inventory, per-character example counts, OCR CER/WER, geometric metrics, confidence intervals where appropriate, model/checkpoint hash, and known failure cases. Never report a single aggregate score without the sample count and split procedure.

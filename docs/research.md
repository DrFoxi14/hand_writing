# Research notes and architecture selection

Research snapshot: 3 November 2026. Re-check upstream repositories, model cards, runtime dependencies, and licenses before adopting third-party code or weights.

## Candidate comparison

| Candidate | What it does | Useful for HIE | Main limitation / decision |
|---|---|---|---|
| **FW-GAN** | Frequency-aware, one-shot handwriting synthesis conditioned on a reference writer; training code and evaluation configs are released. | Strongest near-term candidate for one/few-shot writer-style adaptation. Hugging Face model card states MIT and describes one-shot writer consistency. | Published setup uses Python 3.10 and CUDA 12.6 in its install instructions. Must verify Apple MPS/CPU behavior, exact checkpoint/data provenance, current dependencies, Romanian diacritics, and held-out writer similarity before integration. Treat as an external benchmark candidate, not a drop-in production dependency. |
| **GANwriting** | ECCV 2020 content-conditioned generation of styled handwritten word images; repository declares MIT. | Relevant content/style separation and word-level synthesis. | Repository README describes an older Python 3.7 / PyTorch 1.4 environment. Likely dependency modernization work; use as a research baseline rather than copy legacy stack into core. |
| **SpiS-GAN** | Writer/style-conditioned GAN for handwritten word images, with IAM English and Vietnamese configurations, checkpoints, training and FID/KID utilities. | Useful comparative baseline and evidence for writer-conditioned image generation. | Released artifacts are 32 px with 64 px configs; lexicon-driven word generation. Need test diacritic inventory and resolution/long text behavior for target writing styles. |
| **TrOCR** | Transformer encoder-decoder for handwritten text recognition (OCR), trained on IAM for the handwritten checkpoint. | Possible future optional recognition assistant; it is not the source of truth for math notation or Romanian transcription. | Recognition model, not synthesis. Its model card is IAM-derived and single-line oriented; manual correction is required. |
| **Classical document dewarping** | Geometry from detected text lines and line segments, without a neural model. | Useful design reference for page geometry and curved-page research. | Projective transform only is implemented in the current HIE version. Generalized cylindrical/dewarp estimation remains future work. |

## Selected approach

A hybrid staged system is selected:

- **Now:** classical CV for page and ink analysis; manual labels; verified glyph bank for a transparent, auditable baseline; strict compositing mask.
- **Next:** supervised writer-conditioned neural image generation from line crops, using an explicit text sequence and a reference-line style embedding.
- **Benchmark:** evaluate FW-GAN, then GANwriting/SpiS-GAN where their runtime and language assumptions fit. Do not integrate third-party checkpoints without recording hashes, licenses, and dataset origins.
- **Later:** explicit stroke/vector generation only if it improves held-out similarity and connection geometry over image synthesis. It is not justified to add a recurrent stroke model before collecting trajectory data; photos do not reveal pen-up/pen-down paths or time ordering.

This choice avoids training a high-capacity model on a small number of untranscribed pages, which would mostly memorize paper geometry and page-specific content. It also prevents an ordinary font from masquerading as the writer's actual handwriting.

## Sources

- FW-GAN model card and install/training notes: https://huggingface.co/DAIR-Group/FW_GAN
- GANwriting source repository and MIT license: https://github.com/omni-us/research-GANwriting
- SpiS-GAN model card and released scope: https://huggingface.co/DAIR-Group/SpiS-GAN
- TrOCR handwritten checkpoint card: https://huggingface.co/microsoft/trocr-base-handwritten
- Document dewarping classical-CV reference: https://github.com/PINTO0309/doc-dewarping
- 2019–2024 handwriting synthesis survey, published 2025: https://doi.org/10.1016/j.patcog.2025.111357

## Data requirement reality check

A small collection of untranscribed notebook photographs is not a controlled handwriting corpus. Such samples usually contain a limited and uneven set of Latin letters, Romanian diacritics, numerals, punctuation, formulas, and page positions. They must be independently transcribed and labeled before HIE can infer character coverage or claim generalization.

The code's 40-line minimum is an engineering safety floor that prevents an obviously tiny training run. For an actual personalized model, collect hundreds of accurately transcribed lines across multiple pages and writing sessions, with explicit examples for lowercase/uppercase, Romanian diacritics (ă â î ș ț and uppercase forms), digits, punctuation, mathematical symbols, and contextual letter positions. Reserve complete pages/sessions for validation. Measure performance on new text not present in training.

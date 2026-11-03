# Local dataset storage

This directory is intended for local-only data. Raw images, processed pages, line/glyph crops, annotations, evaluation images, and the SQLite database are intentionally ignored by Git. Keep the `.gitkeep` placeholders, but do not commit personal handwriting data or checkpoints.

Upload the original photographs through the local HIE interface, or use the `hie` ingestion API. Original uploads are stored append-only under `raw/`; derived images are stored under `processed/`.

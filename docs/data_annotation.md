# Annotation guide

1. Upload the original image and wait for rectification.
2. Use the rectified preview, not the camera-space photo, when drawing annotation boxes.
3. For a line crop, include the complete line while minimizing neighboring grid cells and adjacent lines. Enter every character exactly as written, including accents, superscripts, punctuation, and spacing where applicable.
4. Set validation split for selected pages/sessions and never mix crops from a validation page into training.
5. For glyph crops, select one visible character only. Create several examples when the writer's shape changes by context. Use `initial`, `medial`, `final`, and `isolated` where those forms visibly differ; use `any` only when the shape is genuinely context-neutral.
6. Inspect alpha extraction. If a crop includes a neighboring stroke or misses a stroke, do not mark it verified.
7. The UI stores source-page IDs for auditability. Original page files are retained unchanged.

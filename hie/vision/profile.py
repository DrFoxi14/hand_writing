from __future__ import annotations
import cv2
import numpy as np
from collections import Counter
from hie.vision.preprocess import detect_text_lines


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0: return None
    return {"p10": float(np.percentile(values, 10)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "mean": float(np.mean(values))}


def analyze_handwriting(image: np.ndarray, ink_mask: np.ndarray) -> dict:
    if image.shape[:2] != ink_mask.shape[:2]:
        raise ValueError("image and ink_mask dimensions must match")
    n, labels, stats, centers = cv2.connectedComponentsWithStats((ink_mask > 0).astype(np.uint8), 8)
    comps = stats[1:] if n > 1 else np.empty((0,5), dtype=np.int32)
    if len(comps):
        heights, widths, areas = comps[:, cv2.CC_STAT_HEIGHT], comps[:, cv2.CC_STAT_WIDTH], comps[:, cv2.CC_STAT_AREA]
        # Exclude tiny speckles and very large merged words from approximate character statistics.
        keep = (areas >= 5) & (heights >= 4) & (heights <= np.percentile(heights, 90) if len(heights) > 2 else True)
        hstats, wstats = _percentiles(heights[keep]), _percentiles(widths[keep])
    else:
        hstats = wstats = None
    line_boxes = detect_text_lines(ink_mask)
    line_heights = np.array([b[3] for b in line_boxes], dtype=np.float32)
    baselines = np.array([b[1] + b[3] for b in line_boxes], dtype=np.float32)
    line_gaps = np.diff(baselines) if len(baselines) > 1 else np.array([])
    # Estimate dominant stroke slant from connected-component point covariance.
    angles = []
    for x,y,w,h,area in comps:
        if area < 8 or h < 6 or h > image.shape[0] * .12: continue
        ys, xs = np.where(labels[y:y+h, x:x+w] > 0)
        if len(xs) < 8: continue
        cov = np.cov(np.vstack((xs, ys)))
        vals, vecs = np.linalg.eigh(cov)
        vec = vecs[:, np.argmax(vals)]
        angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
        if abs(angle) > 45: angle = angle - 90 if angle > 0 else angle + 90
        angles.append(angle)
    color_pixels = image[ink_mask > 0]
    median_bgr = np.median(color_pixels, axis=0).astype(int).tolist() if len(color_pixels) else None
    ink_density = float((ink_mask > 0).mean())
    return {
        "measurements": {
            "approx_component_height_px": hstats,
            "approx_component_width_px": wstats,
            "line_height_px": _percentiles(line_heights),
            "baseline_gap_px": _percentiles(line_gaps),
            "slant_angle_deg": _percentiles(np.asarray(angles, dtype=np.float32)),
            "ink_density": ink_density,
            "median_ink_bgr": median_bgr,
            "line_count_estimate": len(line_boxes),
            "component_count": int(len(comps)),
        },
        "caveats": [
            "Connected components are not equivalent to individual characters in connected handwriting.",
            "Character proportions and pen pressure need manually labeled glyph/line samples for reliable estimates.",
            "These are photo-space measurements after page rectification, not physical millimetres unless a scale reference is supplied.",
        ],
    }

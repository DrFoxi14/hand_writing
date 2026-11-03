from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np

@dataclass
class PageGeometry:
    corners: list[list[float]]
    rectified_size: list[int]
    detected: bool
    method: str

@dataclass
class PreprocessResult:
    image: np.ndarray
    ink_mask: np.ndarray
    geometry: PageGeometry
    grid_spacing: dict[str, float | None]
    line_boxes: list[list[int]]


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_page_quad(image: np.ndarray) -> tuple[np.ndarray, bool, str]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 25, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (h * w * 0.20):
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4 and area > best_area:
            best, best_area = approx.reshape(4, 2), area
    if best is not None:
        return _order_quad(best), True, "largest_quadrilateral"
    # A safe, explicit fallback: retain the full photograph rather than hallucinating corners.
    full = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    return full, False, "full_frame_fallback"


def rectify_page(image: np.ndarray) -> tuple[np.ndarray, PageGeometry, np.ndarray]:
    quad, detected, method = detect_page_quad(image)
    tl, tr, br, bl = quad
    out_w = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    out_h = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    out_w, out_h = max(2, out_w), max(2, out_h)
    target = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    geometry = PageGeometry(corners=quad.tolist(), rectified_size=[out_w,out_h],
                            detected=detected, method=method)
    return warped, geometry, matrix


def extract_ink_mask(image: np.ndarray) -> np.ndarray:
    """Color-aware extraction for blue and burgundy ink; excludes low-saturation grid lines."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blue = (h >= 88) & (h <= 142) & (s >= 35) & (v <= 245)
    burgundy = ((h <= 12) | (h >= 168)) & (s >= 45) & (v <= 235)
    # Neutral near-black strokes are supported; saturated dark foliage/background is not.
    dark_neutral = (v < 95) & (s < 60)
    mask = (blue | burgundy | dark_neutral).astype(np.uint8) * 255
    # Remove isolated sensor noise but retain fine pen detail.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    return mask


def estimate_grid_spacing(image: np.ndarray) -> dict[str, float | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 11)
    h, w = binary.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 28), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 28)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    row = np.where(horizontal.sum(axis=1) > max(255, w * 0.12))[0]
    col = np.where(vertical.sum(axis=0) > max(255, h * 0.12))[0]
    def spacing(values: np.ndarray) -> float | None:
        if len(values) < 4: return None
        groups = np.split(values, np.where(np.diff(values) > 1)[0] + 1)
        centers = np.array([g.mean() for g in groups if len(g)])
        diffs = np.diff(centers)
        diffs = diffs[(diffs >= 3) & (diffs <= 80)]
        return float(np.median(diffs)) if len(diffs) else None
    return {"horizontal_px": spacing(row), "vertical_px": spacing(col)}


def detect_text_lines(mask: np.ndarray, min_pixels: int = 12) -> list[list[int]]:
    """Return [x,y,w,h] line boxes from row activity; meant for annotation assistance."""
    row_activity = (mask > 0).sum(axis=1)
    threshold = max(min_pixels, int(mask.shape[1] * 0.0018))
    active = row_activity >= threshold
    # Fill tiny vertical gaps inside a text line, but do not bridge normal line spacing.
    active = cv2.morphologyEx(active.astype(np.uint8).reshape(-1,1), cv2.MORPH_CLOSE,
                              np.ones((3,1), np.uint8)).ravel().astype(bool)
    indices = np.where(active)[0]
    if not len(indices): return []
    groups = np.split(indices, np.where(np.diff(indices) > 3)[0] + 1)
    boxes: list[list[int]] = []
    for group in groups:
        if len(group) < 2: continue
        y0, y1 = int(group[0]), int(group[-1])
        local = mask[y0:y1+1]
        cols = np.where((local > 0).sum(axis=0) > 0)[0]
        if not len(cols): continue
        x0, x1 = int(cols[0]), int(cols[-1])
        if (x1 - x0 + 1) >= 5 and (y1 - y0 + 1) >= 3:
            boxes.append([x0, y0, x1-x0+1, y1-y0+1])
    return boxes


def preprocess_array(image: np.ndarray) -> PreprocessResult:
    rectified, geom, _ = rectify_page(image)
    mask = extract_ink_mask(rectified)
    return PreprocessResult(rectified, mask, geom, estimate_grid_spacing(rectified), detect_text_lines(mask))


def preprocess_file(path: Path | str, output_dir: Path | str) -> dict:
    path, output_dir = Path(path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"Could not decode image: {path}")
    result = preprocess_array(image)
    rectified_path = output_dir / "rectified.png"
    mask_path = output_dir / "ink_mask.png"
    if not cv2.imwrite(str(rectified_path), result.image): raise IOError("Failed writing rectified page")
    if not cv2.imwrite(str(mask_path), result.ink_mask): raise IOError("Failed writing ink mask")
    metadata = {"geometry": asdict(result.geometry), "grid_spacing": result.grid_spacing,
                "line_boxes": result.line_boxes, "ink_pixels": int((result.ink_mask > 0).sum()),
                "rectified_path": str(rectified_path), "ink_mask_path": str(mask_path)}
    return metadata

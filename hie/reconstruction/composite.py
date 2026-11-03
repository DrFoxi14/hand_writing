from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def polygon_mask(shape: tuple[int,int], polygon: list[list[float]]) -> np.ndarray:
    h, w = shape
    if len(polygon) < 3: raise ValueError("The authorized region needs at least three polygon points")
    pts = np.asarray(polygon, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("Polygon must be a finite list of [x,y] coordinates")
    if (pts[:,0] < 0).any() or (pts[:,0] >= w).any() or (pts[:,1] < 0).any() or (pts[:,1] >= h).any():
        raise ValueError("Polygon points must lie inside the original photograph")
    if len(pts) == 4:
        pts = order_quadrilateral(pts)
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def order_quadrilateral(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4,2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered,-start,axis=0)
    if abs(cv2.contourArea(ordered.reshape(-1,1,2))) < 4:
        raise ValueError("Quadrilateral area is too small")
    return ordered


def warp_overlay_to_quadrilateral(overlay_rgba: np.ndarray,
                                  polygon: list[list[float]]) -> np.ndarray:
    """Perspective-warp a full-frame overlay into a four-corner writing region.

    The input overlay is expected to contain the rendered text inside the axis-aligned
    bounding box of the selected region. The four polygon points define the target page
    plane. For non-quadrilateral masks, the original overlay is returned unchanged.
    """
    if len(polygon) != 4:
        return overlay_rgba
    h,w = overlay_rgba.shape[:2]
    pts = order_quadrilateral(np.asarray(polygon,dtype=np.float32))
    x0=max(0,int(np.floor(pts[:,0].min()))); y0=max(0,int(np.floor(pts[:,1].min())))
    x1=min(w-1,int(np.ceil(pts[:,0].max()))); y1=min(h-1,int(np.ceil(pts[:,1].max())))
    if x1-x0 < 2 or y1-y0 < 2:
        raise ValueError("Quadrilateral is too small for perspective compositing")
    crop=overlay_rgba[y0:y1+1,x0:x1+1]
    ch,cw=crop.shape[:2]
    source=np.array([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]],dtype=np.float32)
    transform=cv2.getPerspectiveTransform(source,pts)
    warped=cv2.warpPerspective(crop,transform,(w,h),flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    return warped


def composite_authorized_region(original: np.ndarray, overlay_rgba: np.ndarray,
                                polygon: list[list[float]]) -> np.ndarray:
    """Composite alpha only inside the approved polygon; outside pixels are copied byte-for-byte."""
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original must be an RGB image")
    if overlay_rgba.shape[:2] != original.shape[:2] or overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4:
        raise ValueError("overlay must be RGBA and exactly match original dimensions")
    mask = polygon_mask(original.shape[:2], polygon) > 0
    src = original.astype(np.float32)
    over = overlay_rgba[:,:,:3].astype(np.float32)
    alpha = overlay_rgba[:,:,3].astype(np.float32) / 255.0
    alpha = alpha * mask.astype(np.float32)
    out = src.copy()
    out[mask] = (over[mask] * alpha[mask,None] + src[mask] * (1.0-alpha[mask,None]))
    result = np.clip(out,0,255).astype(np.uint8)
    # Assert the invariant at the implementation boundary.
    if not np.array_equal(result[~mask], original[~mask]):
        raise RuntimeError("Pixel-preservation invariant failed")
    return result


def warp_rectified_overlay_to_original(overlay_rgba: np.ndarray, original_shape: tuple[int,int],
                                       original_to_rectified: np.ndarray) -> np.ndarray:
    """Warp a transparent overlay from rectified-page coordinates back to camera coordinates."""
    h,w = original_shape
    inverse = np.linalg.inv(original_to_rectified)
    return cv2.warpPerspective(overlay_rgba, inverse, (w,h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))


def composite_files(original_path: str, overlay_path: str, output_path: str,
                    polygon: list[list[float]]) -> None:
    original = np.asarray(Image.open(original_path).convert("RGB"))
    overlay = np.asarray(Image.open(overlay_path).convert("RGBA"))
    result = composite_authorized_region(original, overlay, polygon)
    Image.fromarray(result).save(output_path)

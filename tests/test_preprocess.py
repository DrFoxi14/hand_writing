import cv2
import numpy as np
from hie.vision.preprocess import extract_ink_mask, estimate_grid_spacing, detect_text_lines, preprocess_array


def synthetic_page():
    im = np.full((600,800,3), 247, np.uint8)
    for x in range(30,800,30): cv2.line(im,(x,0),(x,599),(220,220,220),1)
    for y in range(30,600,30): cv2.line(im,(0,y),(799,y),(220,220,220),1)
    cv2.putText(im,"Sample notes",(75,150),cv2.FONT_HERSHEY_SIMPLEX,1.2,(165,70,30),2,cv2.LINE_AA)
    cv2.putText(im,"Line two",(75,230),cv2.FONT_HERSHEY_SIMPLEX,1.2,(165,70,30),2,cv2.LINE_AA)
    return im


def test_blue_ink_extracted_grid_excluded():
    im = synthetic_page(); mask = extract_ink_mask(im)
    assert mask[145:155,80:160].sum() > 0
    assert mask[30,30] == 0


def test_grid_spacing_and_lines_are_estimated():
    im = synthetic_page(); mask = extract_ink_mask(im)
    spacing = estimate_grid_spacing(im)
    assert spacing["horizontal_px"] is not None
    boxes = detect_text_lines(mask)
    assert len(boxes) >= 2


def test_preprocess_outputs_consistent_dimensions():
    result = preprocess_array(synthetic_page())
    assert result.image.shape[:2] == result.ink_mask.shape
    assert result.geometry.rectified_size[0] > 0
    assert result.line_boxes

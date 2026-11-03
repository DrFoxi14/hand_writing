import numpy as np
import pytest
from hie.reconstruction.composite import (
    composite_authorized_region,
    polygon_mask,
    warp_overlay_to_quadrilateral,
)


def test_pixels_outside_authorized_region_are_identical():
    original=np.full((80,100,3),[10,20,30],dtype=np.uint8)
    overlay=np.zeros((80,100,4),dtype=np.uint8)
    overlay[:,:,:3]=[200,30,80]; overlay[:,:,3]=255
    poly=[[20,15],[75,15],[75,60],[20,60]]
    result=composite_authorized_region(original,overlay,poly)
    mask=polygon_mask(original.shape[:2],poly)>0
    assert np.array_equal(result[~mask],original[~mask])
    assert np.any(result[mask] != original[mask])


def test_perspective_warp_stays_inside_authorized_region_after_composite():
    original=np.full((100,120,3),50,dtype=np.uint8)
    overlay=np.zeros((100,120,4),dtype=np.uint8)
    overlay[25:55,25:75,:3]=[20,80,220]
    overlay[25:55,25:75,3]=255
    # Deliberately unordered corner input; compositor sorts it consistently.
    polygon=[[95,72],[28,78],[18,18],[88,10]]
    warped=warp_overlay_to_quadrilateral(overlay,polygon)
    result=composite_authorized_region(original,warped,polygon)
    mask=polygon_mask(original.shape[:2],polygon)>0
    assert np.any(warped[:,:,3] > 0)
    assert np.array_equal(result[~mask],original[~mask])
    assert np.any(result[mask] != original[mask])


def test_non_quadrilateral_warp_is_identity():
    overlay=np.zeros((30,40,4),dtype=np.uint8)
    polygon=[[2,2],[30,3],[20,25]]
    assert warp_overlay_to_quadrilateral(overlay,polygon) is overlay


def test_invalid_polygon_rejected():
    image=np.zeros((20,20,3),np.uint8); over=np.zeros((20,20,4),np.uint8)
    with pytest.raises(ValueError): composite_authorized_region(image,over,[[1,1],[2,2]])


def test_shape_mismatch_rejected():
    image=np.zeros((20,20,3),np.uint8); over=np.zeros((10,10,4),np.uint8)
    with pytest.raises(ValueError): composite_authorized_region(image,over,[[1,1],[10,1],[10,10],[1,10]])

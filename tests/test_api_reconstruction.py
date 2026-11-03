from io import BytesIO
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from hie.api.main import app


def png_bytes(image: Image.Image) -> bytes:
    bio = BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def test_reconstruction_endpoint_preserves_pixels_outside_polygon():
    original = np.full((90, 120, 3), (232, 230, 224), dtype=np.uint8)
    overlay = np.zeros((90, 120, 4), dtype=np.uint8)
    overlay[20:55, 25:80, :3] = (35, 65, 145)
    overlay[20:55, 25:80, 3] = 255
    polygon = [[18, 15], [100, 8], [106, 72], [14, 78]]
    client = TestClient(app)
    response = client.post(
        "/api/reconstruct",
        files={
            "original": ("page.png", png_bytes(Image.fromarray(original)), "image/png"),
            "overlay": ("overlay.png", png_bytes(Image.fromarray(overlay)), "image/png"),
        },
        data={"polygon": __import__("json").dumps(polygon)},
    )
    assert response.status_code == 200, response.text
    output = np.asarray(Image.open(BytesIO(response.content)).convert("RGB"))
    from hie.reconstruction.composite import polygon_mask
    mask = polygon_mask(original.shape[:2], polygon) > 0
    assert np.array_equal(output[~mask], original[~mask])
    assert np.any(output[mask] != original[mask])

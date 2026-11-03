from __future__ import annotations

def detect_device() -> dict[str, str | bool]:
    """Detect the best available PyTorch device without requiring torch at import time."""
    try:
        import torch
    except Exception:
        return {"device": "cpu", "cuda": False, "mps": False, "torch": False}
    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend and mps_backend.is_available())
    if cuda:
        device = "cuda"
    elif mps:
        device = "mps"
    else:
        device = "cpu"
    return {"device": device, "cuda": cuda, "mps": mps, "torch": True}

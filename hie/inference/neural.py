from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from hie.training.model import WriterConditionedGenerator
from hie.config import CHECKPOINTS


def generate_from_checkpoint(text: str, style_image_path: Path | str,
                             checkpoint_path: Path | str = CHECKPOINTS / "writer_v1.pt",
                             device: str | None = None, pixels_per_char: int = 20) -> Image.Image:
    if not text: raise ValueError("Text cannot be empty")
    checkpoint_path=Path(checkpoint_path)
    if not checkpoint_path.exists(): raise FileNotFoundError(f"No trained checkpoint: {checkpoint_path}")
    ckpt=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    vocab=ckpt.get("vocab",{})
    missing=sorted({ch for ch in text if ch not in vocab})
    if missing: raise ValueError(f"Checkpoint vocabulary does not contain characters: {missing}")
    from hie.device import detect_device
    device=device or detect_device()["device"]
    model=WriterConditionedGenerator(vocab_size=max(vocab.values(),default=0)+1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    style=cv2.imread(str(style_image_path),cv2.IMREAD_GRAYSCALE)
    if style is None: raise ValueError(f"Cannot read style reference: {style_image_path}")
    style=cv2.resize(style,(256,64),interpolation=cv2.INTER_AREA)
    style_tensor=torch.from_numpy(((255.0-style.astype(np.float32))/255.0)[None,None,:,:]).to(device)
    ids=torch.tensor([[vocab[ch] for ch in text]],dtype=torch.long,device=device)
    with torch.inference_mode():
        prediction=model(ids,style_tensor,output_height=64,pixels_per_char=pixels_per_char)[0,0].cpu().numpy()
    alpha=np.clip(prediction*255,0,255).astype(np.uint8)
    # The current network predicts a monochrome ink probability mask; colour can be learned/configured later.
    rgba=np.zeros((alpha.shape[0],alpha.shape[1],4),dtype=np.uint8)
    rgba[:,:,:3]=np.array([39,62,126],dtype=np.uint8)
    rgba[:,:,3]=alpha
    return Image.fromarray(rgba,"RGBA")
